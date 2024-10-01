import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

# https://github.com/opencv/opencv/issues/17687
# https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import csv
import cv2 as cv
import itertools
import json
import math
import numpy as np
import PIL.Image
import PIL.ImageTk
import requests
import time

from tkinter import *
from tkinter import font
from tkinter import ttk
import twisted.internet
from twisted.internet import tksupport, reactor
from twisted.internet import threads as twisted_threads

import puzzbot.camera.camera
from puzzbot.camera.calibrate import CornerDetector, BigHammerCalibrator, BigHammerRemapper

def draw_detected_corners(image, corners, ids = None, *, thickness=1, color=(0,255,255), size=3):
    # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
    shift = 4
    size = size << shift
    for x, y in np.array(corners * (1 << shift), dtype=np.int32):
        cv.rectangle(image, (x-size, y-size), (x+size, y+size), color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
    if ids is not None:
        for (x, y), id in zip(corners, ids):
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

class GantryCalibrator:

    def __init__(self, bot, board, dpi=600.):
        self.bot = bot
        self.board = board
        self.dpi = 600.

    def calibrate(self, coordinates):
        self.move_to(z=0)

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
        
        self.move_to(x=x, y=y)
        time.sleep(.5)

        image = self.bot.get_image()
        corners, ids = detector.detect_corners(image)
        return dict(zip(ids,corners))

    def move_to(self, x=None, y=None, z=None, f=None):
        v = ["G1"]
        if x is not None:
            v.append(f"X{x}")
        if y is not None:
            v.append(f"Y{y}")
        if z is not None:
            v.append(f"Z{z}")
        if f is not None:
            v.append(f"F{f}")
            
        self.bot.send_gcode(' '.join(v))
        self.bot.send_gcode('M400')
    

class AxesStatus(ttk.LabelFrame):

    def __init__(self, parent, axes_and_labels):
        def make_light(color):
            image = PIL.Image.new('RGB', (16,16), color)
            return PIL.ImageTk.PhotoImage(image=image)
        super().__init__(parent, text='Axes')
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

class ScantoolTk:

    def __init__(self, parent, args, config):

        self.server = args.server
        self.config = config
        self.calibration = self.config.get('calibration', {})
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
        root.bind("<<camera>>", self.camera_event)
        self.camera_busy = False
        d = twisted_threads.deferToThread(self.camera.read, 'lores')
        d.addCallback(self.notify_camera_event)

        self.klipper_subscribe()
        poll = twisted.internet.task.LoopingCall(self.klipper_poll)
        poll.start(0)

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

        self.remapper = None
        self.remapper2 = None
        
        calibration = self.config['calibration']
        if 'camera' in calibration:
            self.remapper = BigHammerRemapper.from_calibration(calibration['camera'])
            self.remapper2 = BigHammerRemapper.from_calibration(calibration['camera'], 4)

    def _init_ui(self, parent):
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        w, h = self.camera.frame_size

        self.canvas_camera = Canvas(self.frame, width=w//4, height=h//4,
                                    background='gray', highlightthickness=0)
        self.canvas_camera.grid(column=0, row=0, columnspan=2, sticky=(N, W, E, S))

        self.canvas_detail = Canvas(self.frame, width=w//8, height=h//8,
                                    background='gray', highlightthickness=0)
        self.canvas_detail.grid(column=0, row=1, sticky=(N, W, E, S))

        self.canvas_binary = Canvas(self.frame, width=w//8, height=h//8,
                                    background='gray', highlightthickness=0)
        self.canvas_binary.grid(column=1, row=1, sticky=(N, W, E, S))

        parent.bind("<Key>", self.key_event)

        controls = ttk.Frame(self.frame, padding=5)
        controls.grid(column=0, row=2, columnspan=2, sticky=(N, W, E, S))

        ttk.Button(controls, text='Camera Calibrate', command=self.do_camera_calibrate).grid(column=0, row=0)

        self.var_undistort = IntVar(value=1)
        ttk.Checkbutton(controls, text='Undistort', variable=self.var_undistort).grid(column=1, row=0)

        self.var_detect_corners = IntVar(value=0)
        ttk.Checkbutton(controls, text='Detect Corners', variable=self.var_detect_corners).grid(column=2, row=0)

        self.var_detect_pieces = IntVar(value=0)
        ttk.Checkbutton(controls, text='Detect Pieces', variable=self.var_detect_pieces).grid(column=3, row=0)

        f = ttk.LabelFrame(controls, text='Exposure')
        f.grid(column=4, row=0)
        self.var_exposure = DoubleVar(value=-6)
        Scale(f, from_=-15, to=0, length=200, resolution=.25, orient=HORIZONTAL,
              variable=self.var_exposure, command=self.do_exposure).grid(column=0, row=0)

        self.var_frame_counter = StringVar(value="frame")
        self.frame_no = 0
        self.frame_skip = 0
        ttk.Label(controls, textvar=self.var_frame_counter).grid(column=5, row=0)

        f = ttk.LabelFrame(self.frame, text='GCode')
        f.grid(column=0, row=3, columnspan=2, sticky=(N, W, E, S))
        ttk.Button(f, text='home', command=self.do_home).grid(column=0, row=0)
        ttk.Button(f, text='Gantry Calibrate', command=self.do_gantry_calibrate).grid(column=1, row=0)
        ttk.Button(f, text='gcode1', command=self.do_gcode1).grid(column=2, row=0)
        ttk.Button(f, text='gcode2', command=self.do_gcode2).grid(column=3, row=0)
        ttk.Button(f, text='gcode3', command=self.do_gcode3).grid(column=4, row=0)
        ttk.Button(f, text='gcode4', command=self.do_gcode4).grid(column=5, row=0)

        ttk.Button(f, text='Save Config', command=self.do_save_config).grid(column=6, row=0)

        self.axes_status = AxesStatus(self.frame, [
            ('stepper_x', 'X'),
            ('stepper_y', 'Y1'),
            ('stepper_y1', 'Y2'),
            ('stepper_z', 'Z'),
            ('manual_stepper stepper_a', 'A')
        ])
        self.axes_status.grid(column=0, row=4)

        self.axes_position = AxesPosition(self.frame)
        self.axes_position.grid(column=1, row=4)

    def klipper_poll(self):
        d = twisted_threads.deferToThread(requests.get, self.server + '/bot/notifications', params={'timeout':5})
        d.addCallback(self.klipper_notifications_callback)
        return d

    def klipper_notifications_callback(self, notifications):
        notifications.raise_for_status()
        for o in notifications.json():
            key = o.get('key', None)
            params = o.get('params', None)
            if key is None or params is None or len(o) != 2:
                print("klipper notification mystery:", json.dumps(o, indent=4))
            elif key == 'objects/subscribe':
                self.klipper_objects_subscribe(params)
            elif key == 'gcode/subscribe_output':
                self.klipper_gcode_output(params)
            else:
                print("klipper notification mystery, unknown key:", json.dumps(o, indent=4))

    def klipper_objects_subscribe(self, o):
        eventtime = o.pop('eventtime')
        status = o.pop('status')
        assert len(o) == 0
        
        stepper_enable = status.pop('stepper_enable', None)
        if stepper_enable:
            self.axes_status.set_axes(stepper_enable['steppers'])
            
        motion_report = status.pop('motion_report', None)
        if motion_report:
            X, Y, Z = motion_report['live_position'][:3]
            self.axes_position.set_axes({'X':X, 'Y':Y, 'Z':Z})

        # don't know what to do with this
        # idle_timeout = status.pop('idle_timeout', None)

        if len(status):
            print("klipper_objects_subscribe:", json.dumps(status, indent=4))

    def klipper_gcode_output(self, o):
        print("klipper_gcode_output:", json.dumps(o, indent=4))

    def notify_camera_event(self, image):
        self.image_update = image
        self.frame.event_generate("<<camera>>")
        d = twisted_threads.deferToThread(self.camera.read, 'lores')
        d.addCallback(self.notify_camera_event)

    def do_home(self):
        d = twisted_threads.deferToThread(self.home_impl)

    def home_impl(self):
        print("home!")
        self.send_gcode('G28 Z')
        self.send_gcode('G28 X Y')
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
        calibrator = GantryCalibrator(self, self.charuco_board)
        d = twisted_threads.deferToThread(calibrator.calibrate, coordinates)
        d.addCallback(self.set_gantry_calibration)

    def set_gantry_calibration(self, calibration):
        self.calibration['gantry'] = calibration
            
    def do_gcode1(self):
        print("gcode1!")
        self.send_gcode("G1 Z0")
        self.send_gcode("G1 X0 Y70")

    def do_gcode2(self):
        print("gcode2!")
        self.send_gcode("G1 Z0")

        for x in [0, 70, 140]:
            for y in [70, 140, 210]:
                self.send_gcode(f"G1 X{x} Y{y}")
                self.send_gcode("M400")
                time.sleep(.5)

                path = f'img_{x}_{y}.png'
                print(f"save {x=} {y=} to {path}")
                cv.imwrite(path, self.get_image())
        print("All done!")

    def do_gcode3(self):
        print("gcode3!")
        self.send_gcode("G1 Z0")

        for y in range(475,701,50):
            for x in range(135,676,75):
                self.send_gcode(f"G1 X{x} Y{y} F5000")
                self.send_gcode("M400")
                time.sleep(.5)
                
                path = f'scan_{x}_{y}.png'
                print(f"save {x=} {y=} to {path}")
                cv.imwrite(path, self.get_image())
        print("All done!")
        
    def do_gcode4(self):
        print("gcode4!")

        with open('scan_x/toscan.json') as f:
            coords = json.loads(f.read())

        mm_per_pix = 25.4 / 600
        image_w, image_h = 4000, 3000
        x_offset = -(image_w / 2) * mm_per_pix
        y_offset = (image_h / 2) * mm_per_pix
        
        self.send_gcode("G1 Z0")
        n = len(coords)
        for i, (x, y) in enumerate(coords):
            self.send_gcode(f"G1 X{x+x_offset} Y{y+y_offset} F5000")
            self.send_gcode("M400")
            time.sleep(.5)

            opath = f'scan_x/piece_{i}.png'
            print(f"save piece {i}/{n} to {opath}")
            cv.imwrite(opath, self.get_image())

        print("All done!")
        
    def get_image(self):

        image = self.camera.read()
        
        if self.remapper: # and self.var_undistort.get():
            image = self.remapper.undistort_image(image)
            
        return image

    def klipper_subscribe(self):
        params = {
            'objects': {
                'idle_timeout':['state'],
                'toolhead': ['homed_axes'],
                'motion_report':['live_position'],
                'stepper_enable':None,
                'webhooks':None
            },
            'response_template': {
                'key': 'objects/subscribe'
            }
        }
        o = {'method':'objects/subscribe', 'params':params, 'response':True}
        r = requests.post(self.server + '/bot', json=o)
        print(r.json())
        self.klipper_objects_subscribe(r.json()['result'])
        
        params = {
            'response_template': {
                'key': 'gcode/subscribe_output'
            }
        }
        o = {'method':'gcode/subscribe_output', 'params':params, 'response':True}
        r = requests.post(self.server + '/bot', json=o)
        print(r.json())
        
    def send_gcode(self, script):
        print(f"gcode: {script}")
        o = {'method':'gcode/script', 'params':{'script':script}, 'response':True}
        r = requests.post(self.server + '/bot', json=o)
        o = r.json()
        if o.get('result') != dict():
            print(o)

    def do_exposure(self, arg):
        self.camera.set_exposure(self.var_exposure.get())

    def do_camera_calibrate(self):
        start = time.monotonic()
        print("Calibrating...")
        self.remapper = None
        self.remapper2 = None
        image = self.get_image()

        if image is None:
            print("No image?!")
            return
        
        calibration = BigHammerCalibrator(self.charuco_board).calibrate(image)
        if calibration is not None:
            self.remapper = BigHammerRemapper.from_calibration(calibration)
            self.remapper2 = BigHammerRemapper.from_calibration(calibration, 4)
            
        elapsed = time.monotonic() - start
        print(f"Calibration complete ({elapsed:.3f} seconds)")

        self.calibration['camera'] = calibration

    def key_event(self, event):
        pass

    def camera_event(self, event):

        self.var_frame_counter.set(f"Frame {self.frame_no} ({self.frame_skip} skipped)")
        self.frame_no += 1
        
        if self.camera_busy:
            self.frame_skip += 1
            return

        self.camera_busy = True
        try:
            self.do_camera_event()
        finally:
            self.camera_busy = False

    def do_camera_event(self):
        image_update = self.image_update

        corners, ids = None, None
        
        if self.remapper2 and self.var_undistort.get():
            image_update = self.remapper2.undistort_image(image_update)
                
        if self.var_detect_corners.get():
            corner_detector = CornerDetector(self.charuco_board)
            corners, ids = corner_detector.detect_corners(image_update)

        h, w = image_update.shape[:2]

        if True:
            dst_size = (w, h)
            image_binary = image_camera = image_update.copy()
        else:
            dst_size = (w // 4, h // 4)
            image_binary = image_camera = cv.resize(image_update, dst_size)

        if corners is not None and len(corners) > 0:
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
                draw_detected_corners(image_camera, np.array(list(inside.values())), color=(255,0,0))
            if border:
                draw_detected_corners(image_camera, np.array(list(border.values())), color=(0,255,0))
            if outside:
                draw_detected_corners(image_camera, np.array(list(outside.values())), color=(0,0,255))

        self.update_image_camera(image_camera)
        self.update_image_detail(image_update)
        self.update_image_binary(image_binary)

    def update_image_camera(self, image_camera):
        
        self.image_camera = self.to_photo_image(image_camera)
        self.canvas_camera.delete('all')
        win_w, win_h = self.canvas_camera.winfo_width(), self.canvas_camera.winfo_height()
        img_h, img_w = image_camera.shape[:2]
        x, y = (win_w - img_w) // 2, (win_h - img_h) // 2
        self.canvas_camera.create_image((x, y), image=self.image_camera, anchor=NW)

    def update_image_detail(self, image_full):

        dst_w, dst_h = self.canvas_detail.winfo_width(), self.canvas_detail.winfo_height()
        src_h, src_w = image_full.shape[:2]
        src_x, src_y = (src_w-dst_w)//2, (src_h-dst_h)//2
        image_detail = image_full[src_y:src_y+dst_h, src_x:src_x+dst_w]

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
            thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)[1]
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
        print("Must specific server, either on command line or in configuration")
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
