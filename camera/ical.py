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

def compute_contour_center_of_mass(contour):
    m = cv.moments(contour)
    x = int(m['m10']/m['m00'])
    y = int(m['m01']/m['m00'])
    return (x, y)
    
def choose_contour_center(contour, dpath):

    dpi = 600.
    finger_diameter_mm = 11.7
    
    px_per_mm = dpi / 25.4
    finger_radius_px = px_per_mm * finger_diameter_mm / 2.
    
    center = compute_contour_center_of_mass(contour)

    # as long as we are at least the radius of the finger away from
    # the nearest edge we're happy
    
    # points interior to the polygon have a positive distance
    if cv.pointPolygonTest(contour, center, True) >= finger_radius_px:
        return center

    # center of mass is too close to the edge of the piece, find the
    # point closest to the center of mass isn't too close to an edge
    
    ll = np.min(contour, axis=0) - 5
    ur = np.max(contour, axis=0) + 5
    w, h = ur + 1 - ll

    # construct a new vector of points offset by ll, do _not_ modify
    # the existing points in place, this is purely local bookkeeping
    pp = contour - ll

    piece_image = np.ones((h, w), dtype=np.uint8)
    piece_image[pp[:,1], pp[:,0]] = 0

    piece_image = cv.floodFill(piece_image, None, (0,0), 0)[1]
    
    di = cv.distanceTransform(piece_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    dx2 = np.square(np.arange(w) - (center[0] - ll[0]))
    dy2 = np.square(np.arange(h) - (center[1] - ll[1]))
    di2 = np.sqrt(np.atleast_2d(dx2) + np.atleast_2d(dy2).T)
    assert di.shape == di2.shape

    # derp = cm.afmhot(di2 / np.max(di2))
    # derp = cv.cvtColor(np.uint8(derp * 255), cv.COLOR_RGBA2BGR)
    # derp_path = dpath.replace('_dist.', '_derp.')
    # cv.imwrite(derp_path, derp)

    optimized_center = np.unravel_index(np.argmin(np.where(di >= finger_radius_px, di2, np.inf)), di.shape)
    optimized_center = optimized_center[::-1]

    # if the piece is so small that there are no safe points within it
    # then we get a meaningless answer, make sure we don't blindly
    # accept it
    assert cv.pointPolygonTest(contour, np.float32(optimized_center+ll), True) >= finger_radius_px
    
    finger_okay = np.uint8(np.where(di >= finger_radius_px, 255, 0))
    finger_contours = cv.findContours(finger_okay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    s = 255 / np.max(di)
    di8 = np.array(di * s, dtype=np.uint8)

    di8 = cv.cvtColor(np.uint8(cm.afmhot(di8) * 255), cv.COLOR_RGBA2BGR)
    
    cv.drawContours(di8, [contour-ll], 0, (127,0,127), thickness=2, lineType=cv.LINE_AA)
    cv.drawContours(di8, finger_contours, -1, (0,127,0), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, center-ll, 9, (255, 0, 127), thickness=2, lineType=cv.LINE_AA)
    cv.circle(di8, optimized_center, 9, (127, 255, 0), thickness=2, lineType=cv.LINE_AA)

    print(f"Saving distance to {dpath}")
    cv.imwrite(dpath, di8)

    return optimized_center + ll
    
def get_points_for_piece(ipath, dpath, opath):

    print(f"Reading piece from {ipath}")
    img = cv.imread(ipath, cv.IMREAD_COLOR)
    contour = PerimeterComputer(img, threshold=75).contour
    
    m = cv.moments(contour)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])

    cp = choose_contour_center(np.squeeze(contour), dpath)

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
            for i in (dpath, opath):
                if os.path.exists(i):
                    os.unlink(i)
            get_points_for_piece(ipath, dpath, opath)
            print('-' * 70)

if __name__ == '__main__':
    main()
