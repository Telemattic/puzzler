import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import cv2 as cv
import math
import numpy as np
import puzzler
import re
from puzzler.commands.points import PerimeterComputer
from matplotlib.pylab import cm

def find_polygon_interior_point(polygon):
    for x, y in polygon:
        for i, j in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            test = cv.pointPolygonTest(polygon, (i+0., j+0.), False)
            print(f"  at {(x,y)} on contour, testing {(i,j)} -> {test:+f}")
            if test > 0:
                return (i, j)
    raise ValueException("no interior point in polygon")

def find_center_point(contour, dpath):

    dpi = 600.
    finger_diameter_mm = 11.7
    
    px_per_mm = dpi / 25.4
    finger_radius_px = px_per_mm * finger_diameter_mm / 2.
    
    m = cv.moments(contour)
    m_cx = int(m['m10']/m['m00'])
    m_cy = int(m['m01']/m['m00'])

    pp = np.squeeze(contour)
    ll = np.min(pp, axis=0) - 1
    ur = np.max(pp, axis=0) + 1
    w, h = ur + 1 - ll
    print(f"{ll=} {ur=} {w=} {h=}")

    cols = pp[:,0] - ll[0]
    rows = pp[:,1] - ll[1]

    piece_image = np.ones((h, w), dtype=np.uint8)
    piece_image[rows, cols] = 0
    #cv.imwrite('derp1.png', np.uint8(piece_image*255))
    
    di = cv.distanceTransform(piece_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    # make sure that the seed of the flood fill is inside the contour!
    x, y = w // 2, h // 2
    if cv.pointPolygonTest(pp, (x+ll[0]+0.,y+ll[1]+0.), False) <= 0.:
        print(f"{(x,y)} is not in the polygon, going to try harder I guess...")
        x, y = find_polygon_interior_point(pp) - ll
        print(f"{(x,y)} looks good")
        
    is_exterior_mask = cv.floodFill(piece_image, None, (x,y), 0)[1]
    #derp2 = cv.cvtColor(is_exterior_mask * 255, cv.COLOR_GRAY2BGR)
    #cv.circle(derp2, (x,y), 9, (255,0,127), thickness=2)
    #cv.imwrite('derp2.png', derp2)

    di = np.where(is_exterior_mask, 0., di)

    cy, cx = np.unravel_index(np.argmax(di, axis=None), di.shape)
    c = (cx, cy)
    print(f"find_center_point: {c=} {c+ll=}")

    finger_okay = np.uint8(np.where(di >= finger_radius_px, 255, 0))
    finger_contours = cv.findContours(finger_okay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    mar = cv.minAreaRect(contour)
    r_cx, r_cy = mar[0]
    print(f"minAreaRect={mar}, center: ({r_cx:.1f},{r_cy:.1f})")

    s = 255 / np.max(di)
    di8 = np.array(di * s, dtype=np.uint8)

    if False:
        di8 = cv.cvtColor(di8, cv.COLOR_GRAY2BGR)
    else:
        di8 = cv.cvtColor(np.uint8(cm.afmhot(di8) * 255), cv.COLOR_RGBA2BGR)
    
    cv.drawContours(di8, [contour-ll], 0, (127,0,127), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(di8, finger_contours, -1, (0,127,0), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, c, 9, (255, 0, 127), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, (m_cx, m_cy) - ll, 9, (255, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, np.int32(np.round((r_cx, r_cy) - ll)), 9, (0,255,0), thickness=2, lineType=cv.LINE_AA)

    print(f"Saving distance to {dpath}")
    cv.imwrite(dpath, di8)

    return c+ll
    
def get_points_for_piece(ipath, dpath, opath):

    print(f"Reading piece from {ipath}")
    img = cv.imread(ipath, cv.IMREAD_COLOR)
    contour = PerimeterComputer(img, threshold=75).contour
    
    m = cv.moments(contour)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])

    cp = find_center_point(np.squeeze(contour), dpath)

    print(f"Saving piece to {opath}")
    img = np.copy(np.flip(img, axis=0))
    cv.drawContours(img, [contour], 0, (0,0,255), thickness=2, lineType=cv.LINE_AA)
    cv.circle(img, (cx,cy), 6, (255,255,0), thickness=2, lineType=cv.LINE_AA)
    cv.circle(img, cp, 9, (255, 0, 127), thickness=2, lineType=cv.LINE_AA)
    img = np.copy(np.flip(img, axis=0))

    if os.path.exists(opath):
        os.unlink(opath)
    cv.imwrite(opath, img)

def main():

    for ipath in os.listdir('.'):
        if m := re.match("piece_([A-Z]+[0-9]+)\.jpg", ipath):
            label = m[1]
            dpath = f"piece_{label}_dist.jpg"
            opath = f"piece_{label}_out.jpg"
            get_points_for_piece(ipath, dpath, opath)
            print('-' * 70)

if __name__ == '__main__':
    main()
