import argparse
import numpy as np
import os
import libcamera
import picamera2
import PIL.Image
import PIL.ImageTk
import scipy
import threading
import time

from tkinter import *
from tkinter import font
from tkinter import ttk

# https://github.com/opencv/opencv/issues/17687
# https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv

def make_binary_image(image):
    pass

def find_candidate_pieces(image):
    pass

def find_points_for_piece(subimage):
    pass

def draw_detected_corners(image, corners, ids = None, thickness=1, color=(0,255,255), size=3):
    # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
    shift = 4
    size = size << shift
    for x, y in np.array(np.squeeze(corners) * (1 << shift), dtype=np.int32):
        cv.rectangle(image, (x-size, y-size), (x+size, y+size), color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
    if ids is not None:
        for (x, y), id in zip(np.squeeze(corners), np.squeeze(ids)):
            cv.putText(image, str(id), (int(x)+5, int(y)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

class ICamera:
    pass

class OpenCVCamera(ICamera):

    def __init__(self, camera_id, target_w, target_h):
        print("Opening camera...")
        params = [cv.CAP_PROP_FRAME_WIDTH, target_w, cv.CAP_PROP_FRAME_HEIGHT, target_h]
        cam = cv.VideoCapture(camera_id, cv.CAP_MSMF, params=params)
        assert cam.isOpened()

        def get_frame_size(cam):
            return int(cam.get(cv.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        actual_w, actual_h = get_frame_size(cam)
        print(f"Camera opened: {actual_w}x{actual_h}")

        cam.set(cv.CAP_PROP_FRAME_WIDTH, target_w)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, target_h)

        actual_w, actual_h = get_frame_size(cam)

        print(f"Capture size set to {actual_w}x{actual_h}")

        self._camera = cam
        self._exposure = self.camera.get(cv.CAP_PROP_EXPOSURE)
        self._frame_size = (actual_w, actual_h)

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def exposure(self):
        return self._exposure

    def set_exposure(self, x):
        self._camera.set(cv.CAP_PROP_EXPOSURE, x)
        self._exposure = x

    def read(self):
        ret, frame = self.camera.read()
        assert ret
        return frame

class RPiCamera(ICamera):

    def __init__(self):
        print("Initializing camera...")
        self._camera = picamera2.Picamera2()
        config = self._camera.create_video_configuration(
            main={'size':(4000, 3000), 'format':'RGB888'},
            lores={'size':(640, 480), 'format':'BGR888'},
            transform=libcamera.Transform(hflip=1, vflip=1),
            display='lores')
        self._camera.align_configuration(config)
        print(f"{config=}")
        self._camera.configure(config)

        self._camera.start()
        time.sleep(2)

        actual_w, actual_h = config['main']['size']
        self._frame_size = (actual_w, actual_h)
        
        print(f"Capture size set to {actual_w}x{actual_h}")
        self._exposure = 0.

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def exposure(self):
        return self._exposure

    def set_exposure(self, x):
        self._exposure = x
        # oops! not implemented!

    def read(self):
        # see bricksort/camera/camera_focus.py for fancier request processing
        request = self._camera.capture_request()
        image = None
        try:
            image = request.make_array('main')
        finally:
            request.release()

        return image

class CameraCalibrator:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board

    def get_ij_for_ids(self, ids):
        n_cols, n_rows = self.charuco_board.getChessboardSize()
        return [(k % (n_cols-1), k // (n_cols-1)) for k in ids]

    def fill_missing_corners(self, src_xy, src_ij, min_ij, max_ij):

        have_ij = set(src_ij)
        fill_ij = []
        for i in range(min_ij[0], max_ij[0]):
            for j in range(min_ij[1], max_ij[1]):
                if (i,j) not in have_ij:
                    fill_ij.append((i,j))

        interp = scipy.interpolate.RBFInterpolator(np.array(src_ij), src_xy, neighbors=100)
        fill_xy = interp(np.array(fill_ij))

        return fill_xy, fill_ij

    def find_bounds_ij(self, src_xy, src_ij, image):
    
        image_size = np.array(image.shape[:2][::-1])

        # find the central corner
        dist = np.linalg.norm(src_xy - image_size/2, axis=1)
        central_corner_idx = np.argmin(dist)

        center_ij = src_ij[central_corner_idx]
        center_xy = src_xy[central_corner_idx]

        lookup = {ij: xy for ij, xy in zip(src_ij, src_xy)}
        dists = []
        for (i, j), xy0 in lookup.items():
            xy1 = lookup.get((i-1, j))
            if xy1 is not None:
                dists.append(xy1 - xy0)
            
        src_square_size_px = np.median(np.linalg.norm(dists, axis=1))

        dpi = 600.
        dst_square_size_px = dpi * self.charuco_board.getSquareLength() / 25.4

        print(f"{src_square_size_px=:.1f} {dst_square_size_px=:.1f}")

        print(f"{np.min(src_ij, axis=0)=}")
        print(f"{np.max(src_ij, axis=0)=}")

        min_ij = np.int32(center_ij - np.ceil(center_xy / src_square_size_px))
        max_ij = np.int32(center_ij + np.ceil((image_size - center_xy) / src_square_size_px)) + 1

        min_ij = np.minimum(np.min(src_ij, axis=0), min_ij)
        max_ij = np.maximum(np.max(src_ij, axis=0)+1, max_ij)

        print(f"{min_ij=} {max_ij=}")

        return min_ij, max_ij, center_ij, center_xy, src_square_size_px

    def compute_remaps(self, corners, ids, image):

        src_xy = np.squeeze(corners)
        src_ij = self.get_ij_for_ids(np.squeeze(ids))

        image_h, image_w = image.shape[:2]

        min_ij, max_ij, center_ij, center_xy, square_size_px = self.find_bounds_ij(src_xy, src_ij, image)

        print(f"{square_size_px=:.1f} dpi={2.54*square_size_px:.1f}")

        fill_xy, fill_ij = self.fill_missing_corners(src_xy, src_ij, min_ij, max_ij)

        x, y = fill_xy[:,0], fill_xy[:,1]
        indices = np.argwhere((0 < x) & (x < image_w) & (0 < y) & (y < image_h))

        if len(indices):

            fill_xy[indices] = cv.cornerSubPix(
                image, np.float32(fill_xy[indices]), (11, 11), (-1,-1),
                (cv.TERM_CRITERIA_COUNT+cv.TERM_CRITERIA_EPS, 30, 0.1))

        points_i = np.arange(min_ij[0], max_ij[0])
        points_j = np.arange(min_ij[1], max_ij[1])
        points_x = (points_i - center_ij[0]) * square_size_px + center_xy[0]
        points_y = (points_j - center_ij[1]) * square_size_px + center_xy[1]

        values = np.full((len(points_x), len(points_y), 2), np.nan)

        print(f"{min_ij=} {max_ij=}\n{src_ij=}\n{fill_ij=}")

        for ij, xy in zip(src_ij, src_xy):
            try:
                values[tuple(ij-min_ij)] = xy
            except IndexError:
                print(f"{ij=} {min_ij=} {tuple(ij-min_ij)=} {values.shape=}")
                raise

        for ij, xy in zip(fill_ij, fill_xy):
            values[tuple(ij-min_ij)] = xy

        interp = scipy.interpolate.RegularGridInterpolator((points_x, points_y), values)

        u_range = np.arange(0, image_w, 1)
        v_range = np.arange(0, image_h, 1)
        u, v = np.meshgrid(u_range, v_range)
        u = u.ravel()
        v = v.ravel()
        remaps = interp((u, v))

        assert len(u) == len(v) == len(remaps) == len(u_range)*len(v_range)

        x_map = np.float32(remaps[:,0].reshape((len(v_range), len(u_range))))
        y_map = np.float32(remaps[:,1].reshape((len(v_range), len(u_range))))

        return {'x_map':x_map, 'y_map':y_map, 'src_ij':src_ij, 'src_xy':src_xy, 'fill_ij':fill_ij, 'fill_xy':fill_xy}

    def locate_charuco_corners(self, input_image):
        charuco_dict = self.charuco_board.getDictionary()
        
        charuco_params = cv.aruco.CharucoParameters()
        charuco_params.minMarkers = 0

        detector_params = cv.aruco.DetectorParameters()
        detector_params.cornerRefinementWinSize = 10
        
        detector = cv.aruco.CharucoDetector(
            self.charuco_board, charucoParams=charuco_params, detectorParams=detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(input_image)

        return charuco_corners, charuco_ids
        
    def __call__(self, input_image):
        corners, ids = self.locate_charuco_corners(image)
        return compute_remaps(corners, ids, image)

class ImageRemapper:

    def __init__(self, x_map, y_map):
        self.x_map = x_map
        self.y_map = y_map

    def __call__(self, image, interpolation=cv.INTER_LINEAR):
        return cv.remap(image, self.x_map, self.y_map, interpolation, borderValue=(255,255,0))

class CameraThread(threading.Thread):

    def __init__(self, camera, callback):
        super().__init__(daemon=True)
        self.camera = camera
        self.callback = callback

    def run(self):

        while True:
            frame = self.camera.read()
            self.callback(frame)

class ScantoolTk:

    def __init__(self, parent, camera_id):

        self._init_charuco_board()
        self._init_camera(camera_id, 4656, 3496)
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
        root.bind("<<camera>>", self.camera_event)
        self.camera_thread.start()

    def _init_camera(self, camera_id, target_w, target_h):
        self.camera = None
        if os.path.exists('/dev/video0'):
            self.camera = RPiCamera()
        else:
            self.camera = OpenCVCamera(camera_id, target_w, target_h)

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

        self.var_correct = IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Correct', variable=self.var_correct).grid(column=1, row=0)

        f = ttk.LabelFrame(self.controls, text='Exposure')
        f.grid(column=2, row=0)
        self.var_exposure = IntVar(value=-6)
        Scale(f, from_=-15, to=0, length=200, resolution=1, orient=HORIZONTAL, variable=self.var_exposure, command=self.do_exposure).grid(column=0, row=0)

    def notify_camera_event(self, image):
        self.image_update = image
        self.frame.event_generate("<<camera>>")

    def do_exposure(self, arg):
        self.camera.set(cv.CAP_PROP_EXPOSURE, self.var_exposure.get())

    def do_calibrate(self):
        start = time.monotonic()
        print("Calibrating...")
        image = self.image_update
        self.remapper = None

        if image is None:
            print("No image?!")
            return
        
        calibrator = CameraCalibrator(self.charuco_board)
        charuco_corners, charuco_ids = calibrator.locate_charuco_corners(image)

        if charuco_ids is None or len(charuco_ids) == 0:
            print("Failed to locate corners.")
            return
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        data = calibrator.compute_remaps(charuco_corners, charuco_ids, gray)
        self.remapper = ImageRemapper(data['x_map'], data['y_map'])
        elapsed = time.monotonic() - start
        print(f"Calibration complete ({elapsed:.3f} seconds)")

        image_100 = image.copy()
        draw_detected_corners(image_100, data['src_xy'], ids=data['src_ij'], size=8, thickness=2)
        draw_detected_corners(image_100, data['fill_xy'], ids=data['fill_ij'], color=(255,0,0), size=8, thickness=2)
        cv.imwrite(r'scantool_calibrate_100.png', image_100)
            
        image_25 = cv.resize(image, (1164,874))
        draw_detected_corners(image_25, data['src_xy'] * .25, ids=data['src_ij'])
        draw_detected_corners(image_25, data['fill_xy'] * .25, ids=data['fill_ij'], color=(255,0,0))
        cv.imwrite(r'scantool_calibrate_25.png', image_25)
        
    def key_event(self, event):
        if event.char and event.char in '<>':
            self.exposure += 1 if event.char == '>' else -1
            self.camera.set(cv.CAP_PROP_EXPOSURE, self.exposure)
            print(f"exposure={self.exposure} ({event=})")

    def camera_event(self, event):
        image_update = self.image_update

        if self.remapper is not None and self.var_correct.get():
            image_update = self.remapper(image_update)

        w, h = self.camera.frame_size
        dst_size = (w // 4, h // 4)
        image_camera = cv.resize(image_update, dst_size)

        self.update_image_camera(image_camera)
        self.update_image_detail(image_update)
        self.update_image_binary(image_camera)

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
            y0 = 580
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
    parser.add_argument("-c", "--camera", type=int, default=2)
    args = parser.parse_args()
        
    root = Tk()
    ui = ScantoolTk(root, args.camera)
    root.title("PuZzLeR: scantool")
    root.mainloop()

if __name__ == '__main__':
    main()
