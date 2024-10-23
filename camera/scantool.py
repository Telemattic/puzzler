import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

# https://github.com/opencv/opencv/issues/17687
# https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import collections
import csv
import cv2 as cv
import itertools
import json
import math
import numpy as np
import PIL.Image
import PIL.ImageTk
import pprint
import puzzler
import requests
import scipy.spatial.distance
import time
import traceback

from tkinter import *
from tkinter import font
from tkinter import ttk
import twisted.internet
from twisted.internet import tksupport, reactor
from twisted.internet import threads as twisted_threads

import puzzbot.camera.ffc as ffc
from puzzbot.camera.calibrate import CornerDetector, BigHammerCalibrator, BigHammerRemapper
from puzzler.commands.points import PerimeterComputer

def draw_detected_corners(image, corners, ids = None, *, thickness=1, color=(0,255,255), size=3):
    # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
    shift = 4
    size = size << shift
    for x, y in np.array(corners * (1 << shift), dtype=np.int32):
        ll, ur = (x-size, y-size), (x+size, y+size)
        cv.rectangle(image, ll, ur, color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
    if ids is not None:
        for (x, y), id in zip(corners, ids):
            cv.putText(image, str(id), (int(x)+5, int(y)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

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
        if np.max(di) >= self.finger_radius_px:
            
            h, w = di.shape
            dx2 = np.square(np.arange(w) - (center[0] - self.ll[0]))
            dy2 = np.square(np.arange(h) - (center[1] - self.ll[1]))
            center_di = np.sqrt(np.atleast_2d(dx2) + np.atleast_2d(dy2).T)

            opt_index = np.argmin(np.where(di >= self.finger_radius_px, center_di, np.inf))
        else:
            opt_index = np.argmax(di)

        # reverse the unraveled index to change from (y,x) to (x,y) tuple
        opt_center = np.unravel_index(opt_index, di.shape)[::-1] + self.ll

        # if the piece is so small that there are no safe points within it
        # then we get a meaningless answer, make sure we don't blindly
        # accept it
        dist_to_edge = distance_to_contour(self.contour, opt_center)
        if dist_to_edge < self.finger_radius_px:
            print(f"WARNING: {opt_center=} {dist_to_edge=:.1f}")

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
        
class ImageProjection:

    def __init__(self, coord_mm, px_per_mm):
        self.coord_mm = coord_mm
        self.px_per_mm = px_per_mm

    def px_to_mm(self, px):
        mm_per_px = 1 / self.px_per_mm
        return np.asarray(px) * (mm_per_px, -mm_per_px) + self.coord_mm

    def mm_to_px(self, mm):
        return (np.asarray(mm) - self.coord_mm) * (self.px_per_mm, -self.px_per_mm)

def scan_area_iterator(rect, dxdy):

    # finds an integer number of scans that fits within the rectangle
    # while maintaining the specified stepover, it would be better to
    # reduce the stepover as necessary so that the rectangle is
    # completely covered

    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    dx, dy = dxdy

    c, r = int(math.ceil(w / dx)) + 1, int(math.ceil(h / dy)) + 1

    print(f"scan_area_iterator: {c=} {r=} {dx=:.1f} {dy=:.1f}")
        
    for j in range(r):
        for i in range(c):
            # even rows are left to right, odds rows are right to left
            if j & 1:
                x = x0 + (c-1-i)*dx
            else:
                x = x0 + i*dx
            y = y0 + j*dy
            if x < x0:
                x = x0
            elif x > x1:
                x = x1
            if y < y0:
                y = y0
            elif y > y1:
                y = y1
            yield (x,y)
                
class PieceFinder:

    def __init__(self, gantry, camera):
        self.gantry = gantry
        self.camera = camera
        self.capture_dpi = 600.
        self.panorama_dpi = 75.
        self.image_threshold = 128
        self.min_size_mm = 10
        self.max_size_mm = 70
        self.margin_px = 125
        self.feedrate = 5000
        self.finger_xy = np.array((448.97, 777.38))
        self.finger_diameter_mm = 11.7
        self.board_height = -70.
        self.retract_height = 0.

        #self.image_threshold = 75

    def find_pieces(self, rect):

        print("Finding pieces")

        self.gantry.move_to(z=0)

        dummy_image = self.camera.get_calibrated_image()
        image_h, image_w = dummy_image.shape[:2]

        mm_per_pixel = 25.4 / self.capture_dpi
        x_step = int(image_w * mm_per_pixel)
        y_step = int(image_h * mm_per_pixel)

        print(f"image is {image_w}x{image_h} and {1/mm_per_pixel:.1f} pixels/mm, {x_step=} {y_step=}")
        
        panorama_scale = self.panorama_dpi / self.capture_dpi

        images = []
        for x, y in scan_area_iterator(rect, (x_step, y_step)):
            print(f"Scanning at {x},{y}")
            opath = f"scan_{x:.0f}_{y:.0f}.jpg"
            image = self.capture_image_at_xy(x, y)
            images.append((cv.resize(image, None, fx=panorama_scale, fy=panorama_scale), (x, y)))

        print(f"captured {len(images)} to stitch into panorama")

        pano, proj = self.assemble_panorama(images)

        cv.imwrite('pano.jpg', pano)

        to_scan = self.find_pieces_in_panorama(pano)

        cv.imwrite('pano-contours.png', pano)

        print(f"Found {len(to_scan)} pieces in panorama")

        offset = -np.array((image_w, image_h)) * (self.panorama_dpi / self.capture_dpi) * .5
        
        to_scan = self.convert_px_to_mm(proj, offset, to_scan)

        optimized_path = self.optimize_path(to_scan)

        # with open('pano-tsp.json', 'w') as f:
        #     json.dump({'original':to_scan, 'optimized':optimized_path}, f, indent=4)

        n_cols = int(math.sqrt(len(to_scan)))
        if n_cols * n_cols < len(to_scan):
            n_cols += 1

        pieces = []
        for capture_mm, radius_mm in optimized_path:
            col = len(pieces) % n_cols
            row = len(pieces) // n_cols
            label = f"{chr(ord('A')+row)}{col+1}"
            opath = f"piece_{label}.jpg"
            with np.printoptions(precision=1):
                print(f"\nFound piece at {capture_mm=} {radius_mm=:.1f} mm -> {opath}")
            try:
                points, source, center_mm = self.get_points_and_source_for_piece(capture_mm, radius_mm, opath)
                pieces.append((center_mm, label, points, source))
                
                # self.touch_picture(center_mm, f"piece_{label}_touch.jpg", source.rect)
            except Exception as exc:
                traceback.print_exception(exc)

        Puzzle = puzzler.puzzle.Puzzle

        fnord = []
        for i, (_, label, points, source) in enumerate(pieces):
            fnord.append(Puzzle.Piece(label, source, points, None, None))

        puzzler.file.save('temp-puzzle.json', Puzzle({}, fnord))

        print(f"All done!  Found {len(pieces)} pieces")
        return pieces

    @staticmethod
    def optimize_path(to_scan):
        
        from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search

        vertexes = np.array([xy for xy, _ in to_scan])

        distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vertexes))
        distance_matrix[:,0] = 0

        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

        permutation2, distance2 = solve_tsp_local_search(
            distance_matrix, x0=permutation, perturbation_scheme='ps3')

        return [to_scan[i] for i in permutation2]

    @staticmethod
    def convert_px_to_mm(proj, offset_px, data):

        retval = []
        for center_px, radius_px in data:
            center_mm = proj.px_to_mm(center_px + offset_px)
            radius_mm = radius_px / proj.px_per_mm
            retval.append((center_mm, radius_mm))

        return retval

    def find_pieces_in_panorama(self, image):

        px_per_mm = self.panorama_dpi / 25.4
        min_area = (px_per_mm * self.min_size_mm) ** 2
        max_area = (px_per_mm * self.max_size_mm) ** 2

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, self.image_threshold, 255, cv.THRESH_BINARY)[1]

        contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        print(f"Found {len(contours)} contours!")

        retval = []
        for c in contours:
            
            a = cv.contourArea(c)
            if not (min_area <= a <= max_area):
                continue

            center, radius = cv.minEnclosingCircle(c)
            retval.append((center, radius))

            # draw the identified contour in red
            cv.drawContours(image, [c], 0, (0,0,255))
            
            # draw the bounding circle in yellow, (fractional coordinates, anti-aliasing)
            shift = 4
            scale = 1 << shift
            ix, iy = int(round(center[0] * scale)), int(round(center[1] * scale))
            ir = int(round(radius * scale))
            cv.circle(image, (ix, iy), ir, (0, 255, 255), lineType=cv.LINE_AA, shift=shift)

        return retval

    def assemble_panorama(self, images):
        pano_ul_mm = np.array((min(xy[0] for _, xy in images),
                               max(xy[1] for _, xy in images)))
        px_per_mm = self.panorama_dpi / 25.4

        with np.printoptions(precision=1):
            print(f"{pano_ul_mm=} {px_per_mm=:.3f}")

        proj = ImageProjection(pano_ul_mm, px_per_mm)

        images_px = []
        for i, xy_mm in images:
            xy_px = np.array(proj.mm_to_px(xy_mm), np.int32)
            images_px.append((i, xy_px))

        min_x = min(xy[0] for i, xy in images_px)
        max_x = max(xy[0] + i.shape[1] for i, xy in images_px)
        min_y = min(xy[1] for i, xy in images_px)
        max_y = max(xy[1] + i.shape[0] for i, xy in images_px)

        print(f"{min_x=} {max_x=} {min_y=} {max_y=}")

        pano = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint8)

        print(f"{pano.shape=}")

        for img, (x, y) in images_px:
            x -= min_x
            y -= min_y
            h, w = img.shape[:2]
            pano[y:y+h,x:x+w,:] = img

        return (pano, proj)

    def capture_image_at_xy(self, x, y):
        self.gantry.move_to(x=x, y=y, f=self.feedrate)
        time.sleep(.5)
        return self.camera.get_calibrated_image()

    def get_points_and_source_for_piece(self, capture_xy, radius, opath=None):
        image = self.capture_image_at_xy(capture_xy[0], capture_xy[1])

        image_h, image_w = image.shape[:2]
        pix_per_mm = self.capture_dpi / 25.4
        rect_size = int(radius * pix_per_mm) + self.margin_px
        x0 = max(0, image_w // 2 - rect_size)
        y0 = max(0, image_h // 2 - rect_size)
        x1 = min(image_w, image_w // 2 + rect_size)
        y1 = min(image_h, image_h // 2 + rect_size)
        sub_image = image[y0:y1,x0:x1]

        contour = PerimeterComputer(sub_image, threshold=self.image_threshold).contour

        center = self.choose_contour_center(np.squeeze(contour))

        if opath:
            print(f"Saving piece to {opath}")
            sub_image = np.copy(np.flipud(sub_image))
            cv.drawContours(sub_image, [contour], 0, (0,64,192), thickness=2, lineType=cv.LINE_AA)
            cv.circle(sub_image, center, 6, (64,192,0), thickness=2, lineType=cv.LINE_AA)
            finger_radius_px = int(self.finger_diameter_mm * .5 * pix_per_mm)
            cv.circle(sub_image, center, finger_radius_px, (64,192,0), thickness=2, lineType=cv.LINE_AA)
            sub_image = np.copy(np.flipud(sub_image))
            if os.path.exists(opath):
                os.unlink(opath)
            cv.imwrite(opath, sub_image)

        dx, dy = center + np.array((x0,y0)) - self.finger_xy
        
        dx = center[0] + x0 - self.finger_xy[0]
        dy = y1 - center[1] - self.finger_xy[1]
        
        center_xy = capture_xy + np.array((dx,-dy)) / pix_per_mm

        sub_image_center = np.array(((x1-x0)/2,(y1-y0)/2))
        dxdy_px = center - sub_image_center
        dxdy_mm = dxdy_px / pix_per_mm

        with np.printoptions(precision=2):
            print(f"  sub_image={(x0,y0,x1,y1)} {center=} {center_xy=}")
            print(f"  {sub_image_center=} {dxdy_px=} {dxdy_mm=}")

        Source = puzzler.puzzle.Puzzle.Piece.Source

        # take the center of the points from the center of the
        # image, as that is how the camera was aligned
        points = np.squeeze(contour) - center
        source = Source(opath, (0, 0, sub_image.shape[1], sub_image.shape[0]))
        return points, source, center_xy

    def touch_picture(self, xy, opath, rect):
        self.gantry.move_to(x=xy[0], y=xy[1])
        self.gantry.move_to(z=self.board_height)
        
        time.sleep(.5)
        img = self.camera.get_calibrated_image()
        x0, y0, x1, y1 = rect
        cv.imwrite(opath, img[y0:y1,x0:x1,:])
        
        self.gantry.move_to(z=self.retract_height)

    def choose_contour_center(self, contour):
        px_per_mm = self.capture_dpi / 25.4
        finger_radius_px = px_per_mm * self.finger_diameter_mm / 2.

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
        return cdi.optimize_center(center)
    
class FingerCalibrator:

    def __init__(self, gantry, camera):
        self.gantry = gantry
        self.camera = camera
        self.board = cv.aruco.CharucoBoard(
            (8, 8), 6., 3., cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100))
        self.board_height = -70.
        self.rotate_height = -60

    def calibrate(self, steps=50):

        # motor is 200 steps per revolution
        theta = steps * (math.pi / 100.)
        
        print("Finger calibrate!")

        self.gantry.pump(1)
        time.sleep(10.)
        self.touch()

        image0 = self.camera.get_calibrated_image()
        
        self.rotate(steps)

        image1 = self.camera.get_calibrated_image()

        self.gantry.pump(0)
        self.release()

        center = self.compute_finger_center(theta, image0, image1)

        print("All done!")

        return {'x': center[0], 'y': center[1]}
        
    def calibration_test(self):
        print("Finger calibrate!")

        self.gantry.pump(1)
        time.sleep(20.)
        self.release()

        for steps in range(0,51,5):
        
            self.take_image(f"{steps}_A")
            self.touch()
            self.take_image(f"{steps}_B")
            self.rotate(steps)
            self.take_image(f"{steps}_C")
            self.release()
            self.take_image(f"{steps}_D")

        self.gantry.pump(0)
        print("All done!")

    def touch(self):
        self.gantry.move_to(z=self.board_height)
        time.sleep(.5)

    def rotate(self, angle):
        self.gantry.move_to(z=self.rotate_height)
        self.gantry.rotate(angle)
        self.gantry.move_to(z=self.board_height)
        time.sleep(.5)

    def release(self):
        self.gantry.valve(True)
        time.sleep(2.)
        self.gantry.move_to(z=0)
        self.gantry.valve(False)
        time.sleep(.5)

    def take_image(self, suffix):
        image = self.camera.get_calibrated_image()
        path = f"finger_{suffix}.jpg"
        print(f"  save image to {path}")
        cv.imwrite(path, image)

    def get_corners(self, image):
        cd = CornerDetector(self.board)
        corners, ids = cd.detect_corners(image)
        return dict(zip(ids,corners))

    def compute_finger_center(self, initial_theta, image0, image1):
        corners0 = self.get_corners(image0)
        corners1 = self.get_corners(image1)

        overlap = set(corners0) & set(corners1)

        print(f"{len(overlap)} common corners")

        n_rows = 2 * len(overlap)
        n_cols = 3

        theta = initial_theta
        u_center, v_center = 0., 0.
        for i in range(5):

            theta_deg = theta * 180. / math.pi
            print(f"iteration {i}, {theta=:.5f} rad ({theta_deg:.2f} deg)")

            A = np.zeros((n_rows, n_cols))
            b = np.zeros((n_rows,))

            c = math.cos(theta)
            s = math.sin(theta)

            r = 0
            for i in overlap:
                u0, v0 = corners0[i]
                u1, v1 = corners1[i]

                A[r][0] = -u0*s - v0*c
                A[r][1] = 1
                b[r] = u1 - u0*c + v0*s
                r += 1

                A[r][0] = u0*c - v0*s
                A[r][2] = 1
                b[r] = v1 - u0*s - v0*c
                r += 1

            # print("A=", A)
            # print("b=", b)

            # solve Ax = b
            alpha, x, y = np.linalg.lstsq(A, b, rcond=None)[0]

            alpha_deg = alpha * 180. / math.pi
            print(f"{alpha=:.5f} rad ({alpha_deg:.2f} deg), {x=:.1f} px, {y=:.1f} px")

            theta += alpha

            c = math.cos(theta)
            s = math.sin(theta)

            denom = (1-c)**2 + s**2
            u_center = (x*(1-c) - y*s) / denom
            v_center = (x*s + y*(1-c)) / denom

            print(f"center: ({u_center:.3f},{v_center:.3f})")
            print()

            if math.fabs(alpha) < .001:
                break

        return np.array((u_center, v_center))
        
class RepeatabilityTester:

    def __init__(self, gantry, camera, board):
        self.gantry = gantry
        self.camera = camera
        self.board = board
        self.dpi = 600.

    def test(self, xy0, xy1):

        print("Repeatability Tester!")

        self.gantry.move_to(x=xy0[0], y=xy0[1])
        time.sleep(1)
        image0 = self.camera.get_calibrated_image()
        
        self.gantry.move_to(x=xy1[0], y=xy1[1])
        time.sleep(1)
        image1 = self.camera.get_calibrated_image()

        self.gantry.move_to(x=xy0[0], y=xy0[1])
        time.sleep(1)
        image2 = self.camera.get_calibrated_image()
        
        print("Test 0:")
        offset = self.compute_offset(image0, image1)

        with np.printoptions(precision=2):
            print(f"commanded offset: {xy1-xy0}")
            print(f"observed offset:  {offset * 25.4/self.dpi}")

        print("Test 1:")
        offset = self.compute_offset(image1, image2)

        with np.printoptions(precision=2):
            print(f"commanded offset: {xy0-xy1}")
            print(f"observed offset:  {offset * 25.4/self.dpi}")
            
        print("Test 2:")
        offset = self.compute_offset(image2, image0)

        with np.printoptions(precision=2):
            print(f"commanded offset: {xy0-xy0}")
            print(f"observed offset:  {offset * 25.4/self.dpi}")
            
        print("All done!")
        
    def get_corners(self, image):
        cd = CornerDetector(self.board)
        corners, ids = cd.detect_corners(image)
        return dict(zip(ids,corners))

    def compute_offset(self, image0, image1):
        corners0 = self.get_corners(image0)
        corners1 = self.get_corners(image1)

        overlap = set(corners0) & set(corners1)

        print(f"{len(overlap)} common corners")

        n_rows = 2 * len(overlap)
        n_cols = 3

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows,))

        r = 0
        for i in overlap:
            u0, v0 = corners0[i]
            u1, v1 = corners1[i]

            A[r][0] = -v0
            A[r][1] = 1
            b[r] = u1 - u0
            r += 1

            A[r][0] = u0
            A[r][2] = 1
            b[r] = v1 - v0
            r += 1

        # solve Ax = b
        alpha, x, y = np.linalg.lstsq(A, b, rcond=None)[0]

        alpha_deg = alpha * 180. / math.pi
        print(f"{alpha=:.5f} rad ({alpha_deg:.2f} deg), {x=:.1f} px, {y=:.1f} px")

        return np.array((-x,y))
        
class GantryCalibrator:

    def __init__(self, gantry, camera, board, dpi=600.):
        self.gantry = gantry
        self.camera = camera
        self.board = board
        self.dpi = 600.

    def calibrate(self, coordinates):
        self.gantry.move_to(z=0, cal=False)

        images = {}
        for x, y in coordinates:
            images[(x,y)] = self.find_corners_at(x,y)

        return self.compute_calibration(images)

    def compute_calibration(self, images):
        
        square_length = self.board.getSquareLength()
        board_n_cols, board_n_rows = self.board.getChessboardSize()

        mm_per_pixel = 25.4 / self.dpi

        n_rows = 2 * sum(len(v) for v in images.values())
        n_cols = 4

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows,))

        r = 0
        for (xm, ym), corners in images.items():
            for (ii, jj), (u, v) in corners.items():
                xc = ii * square_length
                yc = (board_n_rows - 1 - jj) * square_length
                A[r,1] = -ym
                A[r,2] = 1
                b[r] = xc - xm - mm_per_pixel * u
                r += 1
                A[r,0] = xm
                A[r,3] = 1
                b[r] = yc - ym + mm_per_pixel * v
                r += 1

        # solve Ax = b
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"{x=}")
        alpha, beta = x[0], x[1]
        error = alpha - beta
        for units, scale in [('radians ', 1.), ('degrees ', 180./math.pi), ('mm/meter', 1000.)]:
            print(f"{units}: alpha={alpha*scale:7.3f} beta={beta*scale:7.3f} error={error*scale:7.3f}")
            
        return {'alpha':x[0], 'beta':x[1]}
    
    def find_corners_at(self, x, y):
            
        detector = CornerDetector(self.board)
        
        self.gantry.move_to(x=x, y=y, cal=False)
        time.sleep(.5)

        image = self.camera.get_calibrated_image()
        corners, ids = detector.detect_corners(image)
        return dict(zip(ids,corners))

class KlipperException(Exception):
    pass

class Klipper:

    def __init__(self, server):
        self.server = server
        self.request_url = self.server + "/bot"
        self.notifications_url = self.server + "/bot/notifications"
        self.callbacks = {}

    def request(self, method, params, response=True):
        o = {'method':method, 'params':params, 'response':response}
        r = requests.post(self.request_url, json=o)
        r.raise_for_status()
        if not response:
            return
        o = r.json()
        if 'result' not in o:
            raise KlipperException(o)
        return o['result']

    def gcode_script(self, script):
        print(f"gcode: {script}")
        o = self.request('gcode/script', {'script':script})
        if o != dict():
            print(o)

    def objects_subscribe(self, callback):
        key = 'objects/subscribe'
        params = {
            'objects': {
                'idle_timeout':['state'],
                'toolhead': ['homed_axes'],
                'motion_report':['live_position'],
                'stepper_enable':None,
                'webhooks':None,
                'output_pin lights':None,
                'output_pin pump':None,
                'output_pin valve':None
            },
            'response_template': {
                'key': key
            }
        }
        self.callbacks[key] = callback
        o = self.request('objects/subscribe', params)
        callback(o)

    def gcode_subscribe_output(self, callback):
        key = 'gcode/subscribe_output'
        params = {
            'response_template': {
                'key': key
            }
        }
        self.callbacks[key] = callback
        return self.request('gcode/subscribe_output', params)
        
    def poll_notifications(self):
        d = twisted_threads.deferToThread(requests.get, self.notifications_url, params={'timeout':5})
        d.addCallback(self._notifications_callback)
        return d

    def _notifications_callback(self, notifications):
        notifications.raise_for_status()
        for o in notifications.json():
            key = o.get('key', None)
            params = o.get('params', None)
            if key is None or params is None or len(o) != 2:
                print("klipper notification mystery:", json.dumps(o, indent=4))
            else:
                self.callbacks[key](params)

class GantryException(Exception):
    pass

class Gantry:

    def __init__(self, klipper, calibration = None):
        self.klipper = klipper
        self.fwd = np.eye(2)
        self.inv = np.eye(2)
        if calibration is not None:
            self.set_calibration(calibration)

    def set_calibration(self, cal):
        alpha, beta = cal['alpha'], cal['beta']
        # fwd: world_coord = self.fwd @ motor_coord.T
        self.fwd = np.array([[np.cos(alpha), -np.sin(beta)],
                             [np.sin(alpha), np.cos(beta)]])
        # inv: motor_coord = self.inv @ world_coord.T
        self.inv = np.linalg.inv(self.fwd)
        
    def move_to(self, x=None, y=None, z=None, f=None, wait=True, cal=True):
        
        if cal:
            if x is not None and y is not None:
                x, y = self.inv @ np.array((x,y)).T
            elif x is not None or y is not None:
                raise GantryException("move_to: calibrated move must specify X and Y")
        
        v = ["G1"]
        if x is not None:
            v.append(f"X{float(x):.2f}")
        if y is not None:
            v.append(f"Y{float(y):.2f}")
        if z is not None:
            v.append(f"Z{float(z):.2f}")
        if f is not None:
            v.append(f"F{f}")
            
        self.klipper.gcode_script(' '.join(v))
        if wait:
            self.klipper.gcode_script('M400')

    def rotate(self, angle, wait=True):
        self.klipper.gcode_script("MANUAL_STEPPER STEPPER=stepper_a SET_POSITION=0")
        self.klipper.gcode_script(f"MANUAL_STEPPER STEPPER=stepper_a MOVE={angle} ACCEL=100")
        if wait:
            self.klipper.gcode_script('M400')

    def lights(self, lights_on):
        value = 1 if lights_on else 0
        self.klipper.gcode_script(f"SET_PIN PIN=lights VALUE={value}")

    def pump(self, pump_on):
        value = 1 if pump_on else 0
        self.klipper.gcode_script(f"SET_PIN PIN=pump VALUE={value}")
            
    def valve(self, open_valve):
        value = 1 if open_valve else 0
        self.klipper.gcode_script(f"SET_PIN PIN=valve VALUE={value}")
    
class CalibratedCamera:

    class IdentityRemapper:
        def undistort_image(self, image):
            return image

    def __init__(self, raw_camera, calibration = None):
        self.raw_camera = raw_camera
        self.remappers = collections.defaultdict(CalibratedCamera.IdentityRemapper)
        if calibration is not None:
            self.set_calibration(calibration)
        self.ffc = None

    def set_calibration(self, calibration):
        self.remappers['main'] = BigHammerRemapper.from_calibration(calibration)
        self.remappers['lores'] = BigHammerRemapper.from_calibration(calibration, 4)

    def get_raw_image(self, stream='main'):
        return self.raw_camera.read(stream)

    def get_raw_image_and_metadata(self, stream='main'):
        return self.raw_camera.read_image_and_metadata(stream)

    def get_calibrated_image(self, stream='main'):
        raw_image = self.get_raw_image(stream)
        image = self.remappers[stream].undistort_image(raw_image)
        return self.ffc(image) if self.ffc else image

    def get_calibrated_image_and_metadata(self, stream='main'):
        raw_image, metadata = self.get_raw_image_and_metadata(stream)
        image = self.remappers[stream].undistort_image(raw_image)
        if self.ffc:
            image = self.ffc(image)
        return (image, metadata)
    
    def get_image(self, stream='main', calibrated=True):
        return self.get_calibrated_image(stream) if calibrated else self.get_raw_image(stream)

    def get_image_and_metadata(self, stream='main', calibrated=True):
        return self.get_calibrated_image_and_metadata(stream) if calibrated else self.get_raw_image_and_metadata(stream)
    
    @property
    def frame_size(self):
        return self.raw_camera.frame_size

class Bot:

    def __init__(self, server, calibration):
        self.server = server
        self.calibration = calibration
        
        self.gantry = Gantry(Klipper(server), self.calibration.get('gantry'))
        
        import puzzbot.camera.camera as pb
        raw_camera = pb.WebCamera(self.server + "/camera")
        
        self.camera = CalibratedCamera(raw_camera, self.calibration.get('camera'))

class AxesHoming(ttk.LabelFrame):

    def __init__(self, parent, axes_and_labels):
        def make_light(color):
            image = PIL.Image.new('RGB', (16,16), color)
            return PIL.ImageTk.PhotoImage(image=image)
        super().__init__(parent, text='Homing')
        self.images = [make_light((200,0,0)), make_light((0,200,0))]
        self.axes = dict()
        for i, (axis, label) in enumerate(axes_and_labels):
            l = ttk.Label(self, image=self.images[0])
            l.grid(column=i*2, row=0, sticky='W')
            self.axes[axis] = l
            l = ttk.Label(self, text=label)
            l.grid(column=i*2+1, row=0, sticky='W')

    def set_axes(self, axes):
        for axis, value in axes.items():
            self.axes[axis].configure(image=self.images[1 if value else 0])

class AxesPosition(ttk.LabelFrame):

    def __init__(self, parent):
        super().__init__(parent, text='Position')
        self.axes = dict()
        for i, axis in enumerate('XYZA'):
            s = StringVar(value=f"{axis} -")
            self.axes[axis] = s
            l = ttk.Label(self, textvar=s, width=12)
            l.grid(column=i, row=0)

    def set_axes(self, axes):
        for axis, value in axes.items():
            self.axes[axis].set(f"{axis}: {value:9.3f}")

class AxesStatus(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        
        self.home_button = ttk.Button(self, text='Home', command=self.do_home)
        self.home_button.grid(column=0, row=0, sticky=(N,S))
        
        self.homing = AxesHoming(self, [
            ('stepper_x', 'X'),
            ('stepper_y', 'Y1'),
            ('stepper_y1', 'Y2'),
            ('stepper_z', 'Z'),
            ('manual_stepper stepper_a', 'A')
        ])
        self.homing.grid(column=1, row=0)

        self.position = AxesPosition(self)
        self.position.grid(column=2, row=0)

    def do_home(self):
        self.event_generate("<<home>>")

class NumberSlider(ttk.Frame):

    def __init__(self, parent, var, options):
        super().__init__(parent)

        self.var = var

        kwargs = {}
        for opt, kw in {'textwidth':'width'}.items():
            if opt in options:
                kwargs[kw] = options[opt]
            
        self.label_format = options.get('format', str)
        self.label = ttk.Label(self, text=self.label_format(self.var.get()), anchor=E, **kwargs)
        self.label.grid(column=0, row=0)

        kwargs = {}
        for kw in ('from_', 'to', 'length', 'orient'):
            if kw in options:
                kwargs[kw] = options[kw]
            
        self.scale = ttk.Scale(self, variable=self.var, **kwargs, command=self.do_scale)
        self.scale.grid(column=1, row=0)

    def do_scale(self, unused):
        self.label['text'] = self.label_format(self.var.get())
        self.event_generate("<<slider>>")

class CameraControls(ttk.LabelFrame):

    def __init__(self, parent):
        super().__init__(parent, text='Camera Controls')

        f = ttk.LabelFrame(self, text='Exposure Time')
        f.grid(column=0, row=0)

        self.var_ae_enable = IntVar(value=1)
        b = ttk.Checkbutton(f, text='Auto Exposure', variable=self.var_ae_enable, command=self.do_ae_enable)
        b.grid(column=0, row=0)

        slider_options = {
            'from_': 1000,
            'to': 30000,
            'length': 200,
            'orient': HORIZONTAL,
            'textwidth': 8
        }

        # full range is 1_000 to 100_000
        self.var_exposure_slider = NumberSlider(f, IntVar(value=6322), slider_options)
        self.var_exposure_slider.grid(column=1, row=0)
        self.var_exposure_slider.bind("<<slider>>", lambda x: self.do_exposure_time())

        f = ttk.LabelFrame(self, text='Threshold')
        f.grid(column=1, row=0)

        slider_options |= {'from_':0, 'to':255}

        self.var_threshold_slider = NumberSlider(f, IntVar(value=128), slider_options)
        self.var_threshold_slider.grid(column=0, row=0)
        self.var_threshold_slider.bind("<<slider>>", lambda x: self.do_threshold())

        f = ttk.LabelFrame(self, text='White Balance')
        f.grid(column=0, row=1, columnspan=2)
        
        self.var_awb_enable = IntVar(value=1)
        b = ttk.Checkbutton(f, text='Auto WB', variable=self.var_awb_enable, command=self.do_awb_enable)
        b.grid(column=0, row=0)

        slider_options |= {'from_':0., 'to':5., 'format': lambda x: f"{x:.4f}", 'textwidth':12}

        # full range is 0. to 32.
        self.var_red_slider = NumberSlider(f, DoubleVar(value=3.2688), slider_options)
        self.var_red_slider.grid(column=1, row=0)
        self.var_red_slider.bind("<<slider>>", lambda x: self.do_colour_gains())
        
        # full range is 0. to 32.
        self.var_blue_slider = NumberSlider(f, DoubleVar(value=1.2043), slider_options)
        self.var_blue_slider.grid(column=2, row=0)
        self.var_blue_slider.bind("<<slider>>", lambda x: self.do_colour_gains())

    @property
    def ae_enable(self):
        return self.var_ae_enable.get() != 0

    @property
    def exposure_time(self):
        return self.var_exposure_slider.var.get()

    @property
    def awb_enable(self):
        return self.var_awb_enable.get() != 0

    @property
    def colour_gains(self):
        return (self.var_red_slider.var.get(), self.var_blue_slider.var.get())

    @property
    def threshold(self):
        return self.var_threshold_slider.var.get()

    def do_ae_enable(self):
        self.event_generate("<<ae_enable>>")

    def do_exposure_time(self):
        self.event_generate("<<exposure_time>>")

    def do_awb_enable(self):
        self.event_generate("<<awb_enable>>")

    def do_colour_gains(self):
        self.event_generate("<<colour_gains>>")

    def do_threshold(self):
        self.event_generate("<<threshold>>")

class ScantoolTk:

    def __init__(self, parent, args, config):

        self.config = config
        self.calibration = self.config.get('calibration', {})
        
        self.gantry = Gantry(Klipper(args.server))
        if 'gantry' in self.calibration:
            self.gantry.set_calibration(self.calibration['gantry'])
        
        self._init_charuco_board(config['charuco_board'])
        self._init_camera(args.camera, 4656, 3496)
        self._init_ui(parent)
        self._init_threads(parent)

    def _init_charuco_board(self, board):

        if board['aruco_dict'] is not None:
            aruco_dict = cv.aruco.getPredefinedDictionary(board['aruco_dict'])
        else:
            dict_path = r'C:\home\eldridge\proj\bricksort\camera\calib\custom_dict_9.json'
            fs = cv.FileStorage(dict_path, cv.FILE_STORAGE_READ)
            aruco_dict = cv.aruco.Dictionary()
            aruco_dict.readDictionary(fs.root())
        
        self.charuco_board = cv.aruco.CharucoBoard(
            board['chessboard_size'],
            board['square_length'],
            board['marker_length'],
            aruco_dict
        )

    def _init_threads(self, root):
        self.background_get_image_and_metadata()

        klipper = self.gantry.klipper
        klipper.objects_subscribe(self.klipper_objects_subscribe_callback)
        klipper.gcode_subscribe_output(self.klipper_gcode_subscribe_output_callback)
        poll = twisted.internet.task.LoopingCall(klipper.poll_notifications)
        poll.start(0)

    def _init_camera(self, camera, target_w, target_h):
        import puzzbot.camera.camera as pb
        self.camera = None
        
        iface = camera[:camera.find(':')]
        if iface.lower() in ('http', 'https'):
            raw_camera = pb.WebCamera(camera)
        elif iface.lower() in ('rpi'):
            raw_camera = pb.RPiCamera()
        elif iface.lower() in ('opencv', 'cv'):
            args = camera[camera.find(':'):]
            camera_id = int(args)
            raw_camera = pb.OpenCVCamera(camera_id, target_w, target_h)
        else:
            raise Exception(f"bad camera scheme: {scheme}")

        self.camera = CalibratedCamera(raw_camera)

        coeffs = np.array(
            [
                [
                    0.8646420515154035,
                    -0.20292869195007895,
                    0.9624986597624194,
                    -1.0081731504881113,
                    0.1565141117932741
                ],
                [
                    0.04042921482042296,
                    5.371389821872379,
                    -18.465929614475783,
                    21.381616567449356,
                    -7.602724632853767
                ],
                [
                    1.2551261867543497,
                    -18.165740801546644,
                    74.40251557645153,
                    -97.83233478066012,
                    39.09967353076903
                ],
                [
                    -2.1595905909907165,
                    23.789326236276644,
                    -98.36057063995587,
                    129.98894019257824,
                    -52.04681934375789
                ],
                [
                    0.9316561999340427,
                    -10.477828331787656,
                    41.39240541911454,
                    -53.10421751042184,
                    20.634210836500802
                ]
            ]
        )
        self.ffc = ffc.FlatFieldCorrector(coeffs)

        calibration = self.config['calibration']
        if 'camera' in calibration:
            self.camera.set_calibration(calibration['camera'])

    def _init_ui(self, parent):
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        w, h = self.camera.frame_size

        self.canvas_camera = Canvas(self.frame, width=w//4, height=h//4,
                                    background='gray', highlightthickness=0)
        self.canvas_camera.grid(column=0, row=0, columnspan=2, sticky=(N, W, E, S))

        parent.bind("<Key>", self.key_event)

        controls = ttk.Frame(self.frame, padding=5)
        controls.grid(column=0, row=2, columnspan=2, sticky=(N, W, E, S))

        ttk.Button(controls, text='Camera Calibrate', command=self.do_camera_calibrate).grid(column=0, row=0)

        self.var_lights = IntVar(value=0)
        ttk.Checkbutton(controls, text='Lights', variable=self.var_lights, command=self.do_lights).grid(column=1, row=0)

        self.var_undistort = IntVar(value=1)
        ttk.Checkbutton(controls, text='Undistort', variable=self.var_undistort).grid(column=2, row=0)

        self.var_detect_corners = IntVar(value=0)
        ttk.Checkbutton(controls, text='Detect Corners', variable=self.var_detect_corners).grid(column=3, row=0)

        self.var_find_contours = IntVar(value=0)
        ttk.Checkbutton(controls, text='Find Contours', variable=self.var_find_contours).grid(column=4, row=0)

        self.var_flat_field_correct = IntVar(value=0)
        ttk.Checkbutton(controls, text='Flat Field Correction',
                        variable=self.var_flat_field_correct,
                        command=self.do_flat_field_correct).grid(column=5, row=0)

        self.var_histogram = IntVar(value=0)
        ttk.Checkbutton(controls, text='Histogram',
                        variable=self.var_histogram).grid(column=6, row=0)
                                                                                                
        ttk.Button(controls, text='Goto', command=self.do_goto).grid(column=7, row=0)
        self.goto_x = IntVar(value=0)
        ttk.Entry(controls, textvar=self.goto_x, width=6).grid(column=8, row=0)
        self.goto_y = IntVar(value=0)
        ttk.Entry(controls, textvar=self.goto_y, width=6).grid(column=9, row=0)
        self.goto_z = IntVar(value=0)
        ttk.Entry(controls, textvar=self.goto_z, width=6).grid(column=10, row=0)

        self.var_pump = IntVar(value=0)
        ttk.Checkbutton(controls, text='Pump', variable=self.var_pump, command=self.do_pump).grid(column=11, row=0)
        
        self.var_valve = IntVar(value=0)
        ttk.Checkbutton(controls, text='Valve', variable=self.var_valve, command=self.do_valve).grid(column=12, row=0)

        f = ttk.LabelFrame(self.frame, text='GCode')
        f.grid(column=0, row=3, columnspan=2, sticky=(N, W, E, S))
        ttk.Button(f, text='Gantry Calibrate', command=self.do_gantry_calibrate).grid(column=0, row=0)
        ttk.Button(f, text='Find Pieces', command=self.do_find_pieces).grid(column=1, row=0)
        ttk.Button(f, text='Finger Calibrate', command=self.do_finger_calibrate).grid(column=2, row=0)
        ttk.Button(f, text='Save Config', command=self.do_save_config).grid(column=3, row=0)
        ttk.Button(f, text='Test Repeatability', command=self.do_test_repeatability).grid(column=4, row=0)
        ttk.Button(f, text='Capture Image', command=self.do_capture_image).grid(column=5, row=0)
        ttk.Button(f, text='Firmware Restart', command=self.do_firmware_restart).grid(column=6, row=0)

        self.axes_status = AxesStatus(self.frame)
        self.axes_status.grid(column=0, row=4, columnspan=2, sticky=(N, W))
        self.axes_status.bind("<<home>>", lambda event: self.do_home())

        self.camera_controls = CameraControls(self.frame)
        self.camera_controls.grid(column=0, row=5, columnspan=2, sticky=(W))
        self.camera_controls.bind("<<ae_enable>>", lambda x: self.do_ae_enable())
        self.camera_controls.bind("<<exposure_time>>", lambda x: self.do_exposure_time())
        self.camera_controls.bind("<<awb_enable>>", lambda x: self.do_awb_enable())
        self.camera_controls.bind("<<colour_gains>>", lambda x: self.do_colour_gains())

    def do_firmware_restart(self):
        self.gantry.klipper.request('gcode/firmware_restart', {})

    def do_capture_image(self):
        img = self.camera.get_calibrated_image()
        cv.imwrite('capture.png', img)

    def do_flat_field_correct(self):
        self.camera.ffc = self.ffc if self.var_flat_field_correct.get() else None

    def do_ae_enable(self):
        cc = self.camera_controls
        cam = self.camera.raw_camera
        if cc.ae_enable:
            cam.set_controls({'AeEnable': cc.ae_enable})
        else:
            cam.set_controls({'ExposureTime': cc.exposure_time, 'AnalogueGain': 2.5})

    def do_exposure_time(self):
        cc = self.camera_controls
        if cc.ae_enable:
            cc.var_ae_enable.set(0)
        else:
            self.do_ae_enable()

    def do_awb_enable(self):
        cc = self.camera_controls
        cam = self.camera.raw_camera
        if cc.awb_enable:
            cam.set_controls({'AwbEnable': cc.awb_enable})
        else:
            cam.set_controls({'ColourGains': cc.colour_gains})

    def do_colour_gains(self):
        cc = self.camera_controls
        if cc.awb_enable:
            cc.var_awb_enable.set(0)
        else:
            self.do_awb_enable()

    def klipper_objects_subscribe_callback(self, o):
        eventtime = o.pop('eventtime')
        status = o.pop('status')
        assert len(o) == 0
        
        stepper_enable = status.pop('stepper_enable', None)
        if stepper_enable:
            self.axes_status.homing.set_axes(stepper_enable['steppers'])
            
        motion_report = status.pop('motion_report', None)
        if motion_report:
            X, Y, Z = motion_report['live_position'][:3]
            self.axes_status.position.set_axes({'X':X, 'Y':Y, 'Z':Z})

        # don't know what to do with this
        idle_timeout = status.pop('idle_timeout', None)

        lights = status.pop('output_pin lights', None)
        if lights:
            self.var_lights.set(1 if lights['value'] != 0. else 0)

        pump = status.pop('output_pin pump', None)
        if pump:
            self.var_pump.set(1 if pump['value'] != 0. else 0)

        valve = status.pop('output_pin valve', None)
        if valve:
            self.var_valve.set(1 if valve['value'] != 0. else 0)

        if len(status):
            print("klipper_objects_subscribe:", json.dumps(status, indent=4))

    def klipper_gcode_subscribe_output_callback(self, o):
        print("klipper_gcode_output:", json.dumps(o, indent=4))

    def background_get_image_and_metadata(self, *args):
        d = twisted_threads.deferToThread(self.camera.get_image_and_metadata, 'lores', self.var_undistort.get() != 0)
        d.addCallback(self.image_update)
        d.addErrback(self.notify_camera_error)
        d.addBoth(self.background_get_image_and_metadata)

    def notify_camera_error(self, arg):
        # twisted.python.failure.Failure
        print(type(arg))
        print(arg)

    def do_goto(self):
        x = self.goto_x.get()
        y = self.goto_y.get()
        z = self.goto_z.get()
        d = twisted_threads.deferToThread(self.gantry.move_to, x=x, y=y, z=z)

    def do_home(self):
        d = twisted_threads.deferToThread(self.home_impl)

    def home_impl(self):
        print("home!")
        self.send_gcode('G28 Z\nG28 X Y')
        # self.send_gcode('G28 X Y')
        self.send_gcode('MANUAL_STEPPER STEPPER=stepper_a ENABLE=1 SET_POSITION=0')
        self.send_gcode('M400')
        # in theory we could do a force move on stepper_y or
        # stepper_y1 to account for non-orthogonality of the X and Y
        # axes after homing
        #
        # self.send_gcode('FORCE_MOVE STEPPER=stepper_y DISTANCE=3 VELOCITY=500')
        print("all done!")

    def do_save_config(self):
        o = self.config.copy()
        o['calibration'] = self.calibration
        with open('the-config.json', 'w') as f:
            json.dump(o, f, indent=4)

    def do_gantry_calibrate(self):
        coordinates = list(itertools.product([0, 70, 140], [70, 140, 210]))
        calibrator = GantryCalibrator(self.gantry, self.camera, self.charuco_board)
        d = twisted_threads.deferToThread(calibrator.calibrate, coordinates)
        d.addCallback(self.set_gantry_calibration)

    def set_gantry_calibration(self, calibration):
        self.calibration['gantry'] = calibration
        self.gantry.set_calibration(calibration)

    def do_finger_calibrate(self):
        calibrator = FingerCalibrator(self.gantry, self.camera)
        d = twisted_threads.deferToThread(calibrator.calibrate)
        d.addCallback(self.set_finger_calibration)

    def set_finger_calibration(self, calibration):
        self.calibration['finger'] = calibration    

    def do_find_pieces(self):
        finder = PieceFinder(self.gantry, self.camera)
        finder.threshold = self.camera_controls.threshold
        rect = (135, 400, 720, 1150)
        rect = (14, 750, 750, 1330)
        d = twisted_threads.deferToThread(finder.find_pieces, rect)
            
    def do_test_repeatability(self):
        tester = RepeatabilityTester(self.gantry, self.camera, self.charuco_board)
        xy0 = np.array((10,100))
        xy1 = xy0 + np.array((46.53, -19.37))
        d = twisted_threads.deferToThread(tester.test, xy0, xy1)

    def send_gcode(self, script):
        self.gantry.klipper.gcode_script(script)

    def do_camera_calibrate(self):
        start = time.monotonic()
        print("Calibrating...")
        image = self.camera.get_raw_image()

        if image is None:
            print("No image?!")
            return
        
        calibration = BigHammerCalibrator(self.charuco_board).calibrate(image)
        if calibration is not None:
            self.camera.set_calibration(calibration)
            
        elapsed = time.monotonic() - start
        print(f"Calibration complete ({elapsed:.3f} seconds)")

        self.calibration['camera'] = calibration

    def do_AeEnable(self):
        controls = {'AeEnable': self.var_AeEnable.get() != 0}
        controls['AwbEnable'] = controls['AeEnable']
        self.camera.raw_camera.set_controls(controls)

    def do_lights(self):
        self.gantry.lights(self.var_lights.get())

    def do_pump(self):
        self.gantry.pump(self.var_pump.get())

    def do_valve(self):
        self.gantry.valve(self.var_valve.get())

    def key_event(self, event):
        pass

    def detect_corners_and_annotate_image(self, src_image, dst_image):

        corner_detector = CornerDetector(self.charuco_board)
        corners, ids = corner_detector.detect_corners(src_image)

        if len(corners) == 0:
            return
        
        min_i, min_j, max_i, max_j = BigHammerCalibrator.maximum_rect(ids)
        inside = dict()
        border = dict()
        outside = dict()
        for ij, uv in zip(ids, corners):
            if min_i <= ij[0] <= max_i and min_j <= ij[1] <= max_j:
                inside[ij] = uv
            elif min_i-1 <= ij[0] <= max_i+1 and min_j-1 <= ij[1] <= max_j+1:
                border[ij] = uv
            else:
                outside[ij] = uv
        if inside:
            draw_detected_corners(dst_image, np.array(list(inside.values())), color=(255,0,0))
        if border:
            draw_detected_corners(dst_image, np.array(list(border.values())), color=(0,255,0))
        if outside:
            draw_detected_corners(dst_image, np.array(list(outside.values())), color=(0,0,255))

    def find_contours_and_annotate_image(self, src_image, dst_image, threshold):

        gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
        h, w = gray.shape

        if False:
            blur0 = cv.medianBlur(gray, 7)
            blur1 = cv.medianBlur(gray, 31)

            xweight = np.sin(np.linspace(0, math.pi, num=w))
            yweight = np.sin(np.linspace(0, math.pi, num=h))
            weight = yweight[:,np.newaxis] @ xweight[np.newaxis,:]

            blur = np.uint8(blur0 * (1. - weight) + blur1 * weight)

            thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)[1]
        else:
            thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)[1]

        contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]

        cv.drawContours(dst_image, contours, -1, (0,0,255))

    def show_histogram(self, src_image, dst_image):

        gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
        hist = np.bincount(gray.ravel(), minlength=256)
        x = np.arange(256, dtype=np.int32) + 16
        y = dst_image.shape[0]-16-np.int32(hist * (256 / np.max(hist)))
        pts = np.vstack((x,y)).T
        cv.polylines(dst_image, [pts], False, (255,0,0))
        cv.putText(dst_image, f"min={np.min(gray)} max={np.max(gray)}",
                   (16,32), cv.FONT_HERSHEY_PLAIN, 1, (0,255,255))
        
    def image_update(self, arg):

        src_image, metadata = arg

        dst_image = None
        
        if self.var_detect_corners.get():
            if dst_image is None: dst_image = src_image.copy()
            self.detect_corners_and_annotate_image(src_image, dst_image)

        if self.var_find_contours.get():
            if dst_image is None: dst_image = src_image.copy()
            self.find_contours_and_annotate_image(src_image, dst_image, self.camera_controls.threshold)

        if self.var_histogram.get():
            if dst_image is None: dst_image = src_image.copy()
            self.show_histogram(src_image, dst_image)

        self.image_camera = self.to_photo_image(dst_image if dst_image is not None else src_image)
        self.canvas_camera.delete('all')
        win_w, win_h = self.canvas_camera.winfo_width(), self.canvas_camera.winfo_height()
        img_h, img_w = src_image.shape[:2]
        x, y = (win_w - img_w) // 2, (win_h - img_h) // 2
        self.canvas_camera.create_image((x, y), image=self.image_camera, anchor=NW)

        ignore_keys = {
            'ColourCorrectionMatrix',
            'FocusFoM',
            'FrameDuration',
            'ScalerCrop',
            'ScalerCrops',
            'SensorBlackLevels',
            'SensorTemperature',
            'SensorTimestamp'
        }

        m = []
        for k, v in metadata.items():
            if k in ignore_keys:
                continue
            if k == 'Lux':
                v = f"{v:.1f}"
            elif k == 'ColourGains':
                v = f"({v[0]:.4f},{v[1]:.4f})"
            elif k == 'DigitalGain':
                v = f"{v:.4f}"
            elif k == 'AnalogueGain':
                v = f"{v:.4f}"
            m.append((k,v))

        s = [f"{k:17s} {v}" for k, v in sorted(m)]
        s = '\n'.join(s)

        self.canvas_camera.create_text((16,16), text=s, fill='LightGreen', anchor='nw', font=('Courier',10))
        
    @staticmethod
    def to_photo_image(image):
        h, w = image.shape[:2]
        if image.ndim == 3:
            dst_mode = 'RGB'
            src_mode = 'BGR'
        else:
            dst_mode = 'L'
            src_mode = 'L'
        image = PIL.Image.frombuffer(dst_mode, (w, h), image.tobytes(), 'raw', src_mode, 0, 1)
        return PIL.ImageTk.PhotoImage(image=image)

def main():
    parser = argparse.ArgumentParser(prog='scantool')
    parser.add_argument("-c", "--config")
    parser.add_argument("--camera")
    parser.add_argument("-s", "--server")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {}

    if args.server:
        config['server'] = args.server

    if 'charuco_board' not in config:
        config['charuco_board'] = {
            'chessboard_size': (33,44),
            'square_length': 6.,
            'marker_length': 3.,
            'aruco_dict':cv.aruco.DICT_4X4_1000
        }

    if 'calibration' not in config:
        config['calibration'] = {}

    if 'server' not in config:
        print("Must specify server, either on command line or in configuration")
        return

    args.server = config['server']
    if args.camera is None:
        args.camera = args.server + "/camera"

    root = Tk()
    ui = ScantoolTk(root, args, config)
    root.title("PuZzLeR: scantool")

    root.protocol('WM_DELETE_WINDOW', reactor.stop)
    
    # root.mainloop()
    tksupport.install(root)
    reactor.run()

if __name__ == '__main__':
    main()
