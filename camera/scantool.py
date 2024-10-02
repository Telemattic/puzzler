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
import requests
import time

from tkinter import *
from tkinter import font
from tkinter import ttk
import twisted.internet
from twisted.internet import tksupport, reactor
from twisted.internet import threads as twisted_threads

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

    def __init__(self, klipper, camera, board, dpi=600.):
        self.klipper = klipper
        self.camera = camera
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

        image = self.camera.get_calibrated_image()
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
            
        self.klipper.gcode_script(' '.join(v))
        self.klipper.gcode_script('M400')
    

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
                'webhooks':None
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

class CalibratedCamera:

    def __init__(self, raw_camera):
        self.raw_camera = raw_camera
        self.remappers = collections.defaultdict(lambda x: x)

    def set_calibration(self, calibration):
        self.remappers['main'] = BigHammerRemapper.from_calibration(calibration)
        self.remappers['lores'] = BigHammerRemapper.from_calibration(calibration, 4)

    def get_raw_image(self, stream='main'):
        return self.raw_camera.read(stream)

    def get_calibrated_image(self, stream='main'):
        raw_image = self.get_raw_image(stream)
        return self.remappers[stream].undistort_image(raw_image)

    def get_image(self, stream='main', calibrated=True):
        return self.get_calibrated_image(stream) if calibrated else self.get_raw_image(stream)

    @property
    def frame_size(self):
        return self.raw_camera.frame_size

class ScantoolTk:

    def __init__(self, parent, args, config):

        self.klipper = Klipper(args.server)
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
        d = twisted_threads.deferToThread(self.camera.get_image, 'lores', self.var_undistort.get() != 0)
        d.addCallback(self.notify_camera_event)

        self.klipper.objects_subscribe(self.klipper_objects_subscribe_callback)
        self.klipper.gcode_subscribe_output(self.klipper_gcode_subscribe_output_callback)
        poll = twisted.internet.task.LoopingCall(self.klipper.poll_notifications)
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

        self.var_undistort = IntVar(value=1)
        ttk.Checkbutton(controls, text='Undistort', variable=self.var_undistort).grid(column=1, row=0)

        self.var_detect_corners = IntVar(value=0)
        ttk.Checkbutton(controls, text='Detect Corners', variable=self.var_detect_corners).grid(column=2, row=0)

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

    def klipper_objects_subscribe_callback(self, o):
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

    def klipper_gcode_subscribe_output_callback(self, o):
        print("klipper_gcode_output:", json.dumps(o, indent=4))

    def notify_camera_event(self, image):
        self.image_update(image)
        d = twisted_threads.deferToThread(self.camera.get_image, 'lores', self.var_undistort.get() != 0)
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
        calibrator = GantryCalibrator(self.klipper, self.camera, self.charuco_board)
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
                cv.imwrite(path, self.camera.get_calibrated_image())
        print("All done!")

    def do_gcode3(self):
        d = twisted_threads.deferToThread(self.gcode3_impl)

    def gcode3_impl(self):
        print("gcode3!")
        self.send_gcode("G1 Z0")

        for y in range(475,701,50):
            for x in range(135,676,75):
                self.send_gcode(f"G1 X{x} Y{y} F5000")
                self.send_gcode("M400")
                time.sleep(.5)
                
                path = f'scan_{x}_{y}.png'
                print(f"save {x=} {y=} to {path}")
                cv.imwrite(path, self.camera.get_calibrated_image())
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
            cv.imwrite(opath, self.get_calibrated_image())

        print("All done!")
        
    def send_gcode(self, script):
        self.klipper.gcode_script(script)

    def do_exposure(self, arg):
        self.camera.set_exposure(self.var_exposure.get())

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

    def key_event(self, event):
        pass

    def detect_corners_and_annotate_image(self, image):

        corner_detector = CornerDetector(self.charuco_board)
        corners, ids = corner_detector.detect_corners(image)

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
            draw_detected_corners(image_camera, np.array(list(inside.values())), color=(255,0,0))
        if border:
            draw_detected_corners(image_camera, np.array(list(border.values())), color=(0,255,0))
        if outside:
            draw_detected_corners(image_camera, np.array(list(outside.values())), color=(0,0,255))

    def image_update(self, image):

        if self.var_detect_corners.get():
            self.detect_corners_and_annotate_image(image)
        
        self.image_camera = self.to_photo_image(image)
        self.canvas_camera.delete('all')
        win_w, win_h = self.canvas_camera.winfo_width(), self.canvas_camera.winfo_height()
        img_h, img_w = image.shape[:2]
        x, y = (win_w - img_w) // 2, (win_h - img_h) // 2
        self.canvas_camera.create_image((x, y), image=self.image_camera, anchor=NW)

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
