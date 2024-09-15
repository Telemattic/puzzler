import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

# https://github.com/opencv/opencv/issues/17687
# https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import csv
import io
import json
import math
import numpy as np
import PIL.Image
import PIL.ImageTk
import requests
import scipy
import threading
import time

from tkinter import *
from tkinter import font
from tkinter import ttk

import cv2 as cv
import puzzbot.camera.camera
from puzzbot.camera.calibrate import CameraCalibrator, PerspectiveComputer, PerspectiveWarper

def draw_detected_corners(image, corners, ids = None, *, thickness=1, color=(0,255,255), size=3):
    # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
    shift = 4
    size = size << shift
    for x, y in np.array(np.squeeze(corners) * (1 << shift), dtype=np.int32):
        cv.rectangle(image, (x-size, y-size), (x+size, y+size), color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
    if ids is not None:
        for (x, y), id in zip(np.squeeze(corners), np.squeeze(ids)):
            cv.putText(image, str(id), (int(x)+5, int(y)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

def draw_homography_points(image, corners, mask):
    shift = 4
    size = 8 << shift
    for (x, y), m in zip(np.array(np.squeeze(corners) * (1 << shift), dtype=np.int32), mask):
        color = (0,255,255) if m else (255,255,0)
        cv.rectangle(image, (x-size, y-size), (x+size, y+size), color,
                     thickness=1, lineType=cv.LINE_AA, shift=shift)

def draw_grid(image):

    h, w = image.shape[:2]
    s = 128
    for x in range(s,w,s):
        cv.polylines(image, [np.array([(x,0), (x,h-1)])], False, (192, 192, 0))
    for y in range(s,h,s):
        cv.polylines(image, [np.array([(0,y), (w-1,y)])], False, (192, 192, 0))

class CameraThread(threading.Thread):

    def __init__(self, camera, callback):
        super().__init__(daemon=True)
        self.camera = camera
        self.callback = callback

    def run(self):

        i = 0
        while True:
            frame = self.camera.read()
            self.callback(frame)
            i += 1

class ScantoolTk:

    def __init__(self, parent, camera):

        self._init_charuco_board()
        self._init_camera(camera, 4656, 3496)
        self._init_ui(parent)
        self._init_threads(parent)

    def _init_charuco_board(self):
        chessboard_size = (16, 12)
        square_length = 10.
        marker_length = 5.
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        self.charuco_board = cv.aruco.CharucoBoard(
            chessboard_size, square_length, marker_length, aruco_dict)
        self.charuco_corners = None
        self.remapper = None

    def _init_threads(self, root):
        self.camera_thread = CameraThread(self.camera, self.notify_camera_event)
        self.camera_busy = False
        root.bind("<<camera>>", self.camera_event)
        self.camera_thread.start()

    def _init_camera(self, camera, target_w, target_h):
        import puzzbot.camera.camera as pb
        self.camera = None
        
        iface = camera[:camera.find(':')]
        if iface.lower() in ('http', 'https'):
            self.camera = pb.WebCamera(camera)
        elif iface.lower() in ('rpi'):
            self.camera = pb.RPiCamera()
        elif iface.lower() in ('opencv', 'cv'):
            args = camera[camera.find(':'):]
            camera_id = int(args)
            self.camera = pb.OpenCVCamera(camera_id, target_w, target_h)
        else:
            raise Exception(f"bad camera scheme: {scheme}")

    def _init_ui(self, parent):
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        w, h = self.camera.frame_size

        self.canvas_camera = Canvas(self.frame, width=w//4, height=h//4,
                                    background='white', highlightthickness=0)
        self.canvas_camera.grid(column=0, row=0, columnspan=2, sticky=(N, W, E, S))

        self.canvas_detail = Canvas(self.frame, width=w//8, height=h//8,
                                    background='white', highlightthickness=0)
        self.canvas_detail.grid(column=0, row=1, sticky=(N, W, E, S))

        self.canvas_binary = Canvas(self.frame, width=w//8, height=h//8,
                                    background='white', highlightthickness=0)
        self.canvas_binary.grid(column=1, row=1, sticky=(N, W, E, S))

        parent.bind("<Key>", self.key_event)

        self.controls = ttk.Frame(self.frame, padding=5)
        self.controls.grid(column=0, row=2, columnspan=2, sticky=(N, W, E, S))

        ttk.Button(self.controls, text='Calibrate', command=self.do_calibrate).grid(column=0, row=0)

        self.var_undistort = IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Undistort', variable=self.var_undistort).grid(column=1, row=0)

        self.var_detect_corners = IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Detect Corners', variable=self.var_detect_corners).grid(column=2, row=0)

        self.var_fix_perspective = IntVar(value=0)
        ttk.Checkbutton(self.controls, text='Fix Perspective', variable=self.var_fix_perspective).grid(column=3, row=0)

        self.var_detect_pieces = IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Detect Pieces', variable=self.var_detect_pieces).grid(column=4, row=0)

        f = ttk.LabelFrame(self.controls, text='Exposure')
        f.grid(column=5, row=0)
        self.var_exposure = DoubleVar(value=-6)
        Scale(f, from_=-15, to=0, length=200, resolution=.25, orient=HORIZONTAL, variable=self.var_exposure, command=self.do_exposure).grid(column=0, row=0)

        self.var_pitch_yaw_roll = StringVar(value="pyr")
        ttk.Label(self.controls, textvar=self.var_pitch_yaw_roll).grid(column=6, row=0)

        self.var_frame_counter = StringVar(value="frame")
        self.frame_no = 0
        ttk.Label(self.controls, textvar=self.var_frame_counter).grid(column=7, row=0, sticky='E')

    def notify_camera_event(self, image):
        self.image_update = image
        self.frame.event_generate("<<camera>>")

    def do_exposure(self, arg):
        self.camera.set_exposure(self.var_exposure.get())

    def do_calibrate(self):
        start = time.monotonic()
        print("Calibrating...")
        image = self.image_update
        self.remapper = None

        if image is None:
            print("No image?!")
            return
        
        calibrator = CameraCalibrator(self.charuco_board)
        charuco_corners, charuco_ids = calibrator.detect_corners(image)

        if charuco_ids is None or len(charuco_ids) == 0:
            print("Failed to locate corners.")
            return

        self.remapper = calibrator.calibrate_camera(charuco_corners, charuco_ids, image)
            
        elapsed = time.monotonic() - start
        print(f"Calibration complete ({elapsed:.3f} seconds)")

    def key_event(self, event):
        if event.char and event.char in '<>':
            self.exposure += .25 if event.char == '>' else -.25
            self.camera.set_exposure(self.exposure)
            print(f"exposure={self.exposure} ({event=})")

    def camera_event(self, event):

        self.var_frame_counter.set(f"Frame {self.frame_no}")
        self.frame_no += 1
        
        if self.camera_busy:
            return

        self.camera_busy = True
        try:
            self.do_camera_event()
        finally:
            self.camera_busy = False

    def do_camera_event(self):
        image_update = self.image_update

        corners, ids = None, None
        
        if self.remapper is not None and self.var_undistort.get():
            image_update = self.remapper.undistort_image(image_update)
        elif self.var_detect_corners.get():
            calibrator = CameraCalibrator(self.charuco_board)
            corners, ids = calibrator.detect_corners(image_update)

        w, h = self.camera.frame_size
        dst_size = (w // 4, h // 4)
        image_binary = image_camera = cv.resize(image_update, dst_size)

        if corners is not None:
            draw_detected_corners(image_camera, corners * .25)

        if corners is not None and self.remapper is not None:
            calibrator = CameraCalibrator(self.charuco_board)
            object_points = calibrator.get_object_points_for_ij(ids)
            camera_matrix = self.remapper.camera_matrix
            dist_coeffs = self.remapper.dist_coeffs
            angles = calibrator.get_euler_angles(object_points, corners, camera_matrix, dist_coeffs)
            if angles is not None:
                pitch, yaw, roll = angles
                self.var_pitch_yaw_roll.set(f"{pitch=:6.2f} {yaw=:6.2f} {roll=:6.2f}")
            else:
                self.var_pitch_yaw_roll.set("pitch= yaw= roll=")

        if self.var_fix_perspective.get() and self.remapper is not None:
            self.var_fix_perspective.set(0)
            self.fix_perspective2(image_update)

        if self.var_fix_perspective.get() and self.remapper is not None:
            self.var_fix_perspective.set(0)
            self.fix_perspective(image_camera)

        self.update_image_camera(image_camera)
        self.update_image_detail(image_update)
        self.update_image_binary(image_binary)

    def fix_perspective(self, image):
        camera_matrix = self.remapper.camera_matrix
        
        R1 = cv.Rodrigues(self.remapper.rvec)[0]
        t1 = self.remapper.tvec

        R2 = np.eye(3)
        t2 = self.remapper.tvec

        with np.printoptions(precision=3):
            print(f"{R1=}")
            print(f"{t1=}")
            print(f"{R2=}")
            print(f"{t2=}")

        # computeC2MC1
        R_1to2 = R2 @ R1.T
        t_1to2 = R2 @ (-R1.T @ t1) + t2

        with np.printoptions(precision=3):
            print(f"{R_1to2=}")
            print(f"{t_1to2=}")

        normal1 = R1 @ np.array((0, 0, 1)).reshape(3,1)
        origin = np.zeros((3,1))
        origin1 = R1 @ origin + t1
        d_inv1 = 1. / np.squeeze(normal1).dot(np.squeeze(origin1))

        with np.printoptions(precision=3):
            print(f"{normal1=}")
            print(f"{origin1=}")
            print(f"{d_inv1=}")

        # computeHomography
        homography_euclidean = R_1to2 + d_inv1 * t_1to2 * normal1.T
        homography = camera_matrix @ homography_euclidean @ np.linalg.inv(camera_matrix)

        with np.printoptions(precision=3):
            print(f"{homography_euclidean=}")
            print(f"{homography=}")

        homography_euclidean /= homography_euclidean[2,2]
        homography /= homography[2,2]
        
        with np.printoptions(precision=3):
            print(f"normalized {homography_euclidean=}")
            print(f"normalized {homography=}")

        image_size = (image.shape[1], image.shape[0])
        warped = cv.warpPerspective(image, homography, image_size)

        image = image.copy()
        draw_grid(image)
        cv.imwrite('scantool_A.png', image)

        draw_grid(warped)
        cv.imwrite('scantool_B.png', warped)

    def fix_perspective2(self, image):

        p = PerspectiveComputer(self.charuco_board).compute_homography(image)

        warper = PerspectiveWarper(p['homography'], p['image_size'])

        warped_image = warper.warp_image(image)

        src_points = p['corners']
        corner_ids = p['corner_ids']
        mask = p['mask']

        image = image.copy()
        draw_grid(image)
        draw_homography_points(image, src_points, mask)
        cv.imwrite('scantool_A.png', image)

        warped_points = warper.warp_points(src_points)
        draw_grid(warped_image)
        draw_homography_points(warped_image, warped_points, mask)
        cv.imwrite('scantool_B.png', warped_image)

        with open('distances.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, 'i j axis dist'.split())
            writer.writeheader()
            lookup = {ij: xy for ij, xy in zip(corner_ids, warped_points)}
            for (i, j), xy0 in lookup.items():
                xy1 = lookup.get((i+1, j))
                if xy1 is not None:
                    dist = np.linalg.norm(xy1-xy0)
                    writer.writerow({'i':i+.5, 'j':j, 'axis':'x', 'dist':dist})
                xy1 = lookup.get((i, j+1))
                if xy1 is not None:
                    dist = np.linalg.norm(xy1-xy0)
                    writer.writerow({'i':i, 'j':j+.5, 'axis':'y', 'dist':dist})
        
    def update_image_camera(self, image_camera):
        
        self.image_camera = self.to_photo_image(image_camera)
        self.canvas_camera.delete('all')
        self.canvas_camera.create_image((0,0), image=self.image_camera, anchor=NW)

    def update_image_detail(self, image_full):

        src_h, src_w, _ = image_full.shape
        dst_w, dst_h = (src_w//8, src_h//8)
        src_x, src_y = (src_w-dst_w)//2, (src_h-dst_h)//2
        image_detail = image_full[src_y:src_y+dst_h, src_x:src_x+dst_w]

        if self.charuco_corners is not None:
            self.draw_detected_corners(image_detail, self.charuco_corners - (src_x, src_y))
            
        self.image_detail = self.to_photo_image(image_detail)
        self.canvas_detail.delete('all')
        self.canvas_detail.create_image((0,0), image=self.image_detail, anchor=NW)

    def update_image_binary(self, image_camera):

        if not self.var_detect_pieces.get():
            self.canvas_binary.delete('all')
            return

        gray = cv.cvtColor(image_camera, cv.COLOR_BGR2GRAY)
        hist = np.bincount(image_camera.ravel())

        if False:
            thresh = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, -16)
        elif True:
            thresh = cv.threshold(gray, 84, 255, cv.THRESH_BINARY)[1]
        else:
            thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
            
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
        thresh = cv.erode(thresh, kernel)
        thresh = cv.dilate(thresh, kernel)

        image_binary = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        
        if True:
            contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            cv.drawContours(image_binary, contours, -1, (0,255,0), thickness=2, lineType=cv.LINE_AA)
            for c in contours:
                r = cv.boundingRect(c)
                r = (r[0]-25, r[1]-25, r[2]+50, r[3]+50)
                if r[0] < 0 or r[1] < 0:
                    continue
                if r[0] + r[2] >= thresh.shape[1] or r[1] + r[3] >= thresh.shape[0]:
                    continue
                cv.rectangle(image_binary, r, (255,0,0), thickness=2)

        if True:
            x0 = 20
            y0 = image_binary.shape[0] - 20
            x_coords = x0 + np.arange(len(hist))
            y_coords = y0 - (hist * 200) // np.max(hist)
            pts = np.array(np.vstack((x_coords, y_coords)).T, dtype=np.int32)
            cv.polylines(image_binary, [np.array([(x0, y0), (x0+255, y0)])], False, (192, 192, 0), thickness=2)
            cv.polylines(image_binary, [pts], False, (255, 255, 0), thickness=2)
        
        dst_size = (self.camera.frame_size[0]//8, self.camera.frame_size[1]//8)
        self.image_binary = self.to_photo_image(cv.resize(image_binary, dst_size))
        self.canvas_binary.delete('all')
        self.canvas_binary.create_image((0,0), image=self.image_binary, anchor=NW)

    @staticmethod
    def to_photo_image(image):
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            dst_mode = 'RGB'
            src_mode = 'BGR'
        else:
            dst_mode = 'L'
            src_mode = 'L'
        image = PIL.Image.frombuffer(dst_mode, (w, h), image.tobytes(), 'raw', src_mode, 0, 1)
        return PIL.ImageTk.PhotoImage(image=image)

def main():
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
    sys.path.insert(0, lib)
    
    import puzzbot.camera.camera
    
    parser = argparse.ArgumentParser(prog='scantool')
    parser.add_argument("-c", "--camera", required=True)
    args = parser.parse_args()
        
    root = Tk()
    ui = ScantoolTk(root, args.camera)
    root.title("PuZzLeR: scantool")
    root.mainloop()

if __name__ == '__main__':
    main()
