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

def distance_to_contour(contour, pt):
    return cv.pointPolygonTest(contour, np.float32(pt), True)
    
class ContourDistanceImage:

    def __init__(self, contour, finger_radius_px):
        self.contour = contour
        self.finger_radius_px = finger_radius_px
        
        self.ll = np.min(contour, axis=0) - 5
        self.ur = np.max(contour, axis=0) + 5
        w, h = self.ur + 1 - self.ll

        # construct a new vector of points offset by ll, do _not_ modify
        # the existing points in place, this is purely local bookkeeping
        pp = self.contour - self.ll

        piece_image = np.ones((h, w), dtype=np.uint8)
        piece_image[pp[:,1], pp[:,0]] = 0

        piece_image = cv.floodFill(piece_image, None, (0,0), 0)[1]

        self.di = cv.distanceTransform(piece_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    def optimize_center(self, center):
        di = self.di
        h, w = di.shape
        dx2 = np.square(np.arange(w) - (center[0] - self.ll[0]))
        dy2 = np.square(np.arange(h) - (center[1] - self.ll[1]))
        center_di = np.sqrt(np.atleast_2d(dx2) + np.atleast_2d(dy2).T)

        opt_index = np.argmin(np.where(di >= self.finger_radius_px, center_di, np.inf))

        # reverse the unraveled index to change from (y,x) to (x,y) tuple
        opt_center = np.unravel_index(opt_index, di.shape)[::-1] + self.ll

        # if the piece is so small that there are no safe points within it
        # then we get a meaningless answer, make sure we don't blindly
        # accept it
        dist_to_edge = distance_to_contour(self.contour, opt_center)
        assert dist_to_edge >= self.finger_radius_px

        return opt_center

    def make_color_image(self, center, optimized_center):

        di, ll = self.di, self.ll

        finger_okay = np.uint8(np.where(di >= self.finger_radius_px, 255, 0))
        finger_contours = cv.findContours(finger_okay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

        img = np.uint8(di * (255 / np.max(di)))
        img = cv.cvtColor(np.uint8(cm.afmhot(img) * 255), cv.COLOR_RGBA2BGR)

        cv.drawContours(img, [self.contour-ll], 0, (127,0,127), thickness=2, lineType=cv.LINE_AA)
        cv.drawContours(img, finger_contours, -1, (0,127,0), thickness=2, lineType=cv.LINE_AA)
        if center is not None:
            cv.circle(img, center-ll, 9, (255, 0, 127), thickness=2, lineType=cv.LINE_AA)
        if optimized_center is not None:
            cv.circle(img, optimized_center-ll, 9, (127, 255, 0), thickness=2, lineType=cv.LINE_AA)

        return img
        
def choose_contour_center(contour, dpath = None):

    dpi = 600.
    finger_diameter_mm = 11.7
    
    px_per_mm = dpi / 25.4
    finger_radius_px = px_per_mm * finger_diameter_mm / 2.
    
    center = compute_contour_center_of_mass(contour)

    # as long as we are at least the radius of the finger away from
    # the nearest edge we're happy
    
    # points interior to the polygon have a positive distance
    if distance_to_contour(contour, center) >= finger_radius_px:
        return center

    # center of mass is too close to the edge of the piece, find the
    # point closest to the center of mass that isn't too close to an
    # edge

    cdi = ContourDistanceImage(contour, finger_radius_px)

    optimized_center = cdi.optimize_center(center)

    if dpath:
        img = cdi.make_color_image(center, optimized_center)
        print(f"Saving distance to {dpath}")
        cv.imwrite(dpath, img)

    return optimized_center
    
def get_points_for_piece(ipath, dpath, opath):

    print(f"Reading piece from {ipath}")
    img = cv.imread(ipath, cv.IMREAD_COLOR)
    contour = PerimeterComputer(img, threshold=75).contour
    
    cp = choose_contour_center(np.squeeze(contour), dpath)

    print(f"Saving piece to {opath}")
    img = np.copy(np.flip(img, axis=0))
    cv.drawContours(img, [contour], 0, (0,0,255), thickness=2, lineType=cv.LINE_AA)
    cv.circle(img, cp, 9, (127, 255, 0), thickness=2, lineType=cv.LINE_AA)
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
                    os.remove(i)
            get_points_for_piece(ipath, dpath, opath)
            print('-' * 70)

if __name__ == '__main__':
    main()
