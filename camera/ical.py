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

def find_center_point(contour, dpath):

    dpi = 600.
    finger_diameter_mm = 11.7
    
    px_per_mm = dpi / 25.4
    finger_radius_px = px_per_mm * finger_diameter_mm / 2.
    
    pp = np.squeeze(contour)
    ll = np.min(pp, axis=0) - 5
    ur = np.max(pp, axis=0) + 5
    w, h = ur + 1 - ll
    print(f"{ll=} {ur=} {w=} {h=}")

    # center-of-mass in image coordinates
    m = cv.moments(contour)
    m_cx = int(m['m10']/m['m00']) - ll[0]
    m_cy = int(m['m01']/m['m00']) - ll[1]

    cols = pp[:,0] - ll[0]
    rows = pp[:,1] - ll[1]

    piece_image = np.ones((h, w), dtype=np.uint8)
    piece_image[rows, cols] = 0

    piece_image = cv.floodFill(piece_image, None, (0,0), 0)[1]
    
    di = cv.distanceTransform(piece_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    f_cx, f_cy = None, None
    if di[m_cy,m_cx] < finger_radius_px:
        # center-of-mass is too close to the edge of the piece, find
        # the point closest to the center of mass that is workable
        print("*** center of mass not within 'safe' zone, trying to improve it!")
        dx2 = np.square(np.arange(w) - m_cx)
        dy2 = np.square(np.arange(h) - m_cy)
        di2 = np.sqrt(np.atleast_2d(dx2) + np.atleast_2d(dy2).T)
        assert di.shape == di2.shape
        f_cy, f_cx = np.unravel_index(np.argmin(np.where(di >= finger_radius_px, di2, np.inf)), di.shape)
        print(f"best finger location: {(f_cx,f_cy)}")

    cy, cx = np.unravel_index(np.argmax(di, axis=None), di.shape)
    c = (cx, cy)
    print(f"find_center_point: {c=} {c+ll=}")

    finger_okay = np.uint8(np.where(di >= finger_radius_px, 255, 0))
    finger_contours = cv.findContours(finger_okay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    s = 255 / np.max(di)
    di8 = np.array(di * s, dtype=np.uint8)

    if False:
        di8 = cv.cvtColor(di8, cv.COLOR_GRAY2BGR)
    else:
        di8 = cv.cvtColor(np.uint8(cm.afmhot(di8) * 255), cv.COLOR_RGBA2BGR)
    
    cv.drawContours(di8, [contour-ll], 0, (127,0,127), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(di8, finger_contours, -1, (0,127,0), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, c, 9, (255, 0, 127), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, (m_cx, m_cy), 9, (255, 255, 0), thickness=2, lineType=cv.LINE_AA)
    if f_cx is not None:
        cv.circle(di8, (f_cx, f_cy), 9, (0,255,0), thickness=2, lineType=cv.LINE_AA)

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
