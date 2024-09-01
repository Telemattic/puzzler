import argparse
import cv2 as cv
import numpy as np
import PIL.Image
import PIL.ImageTk
import scipy
import threading

from tkinter import *
from tkinter import font
from tkinter import ttk

def make_binary_image(image):
    pass

def find_candidate_pieces(image):
    pass

def find_points_for_piece(subimage):
    pass

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
    
    def compute_remaps(self, corners, ids, image_shape):

        src_xy = np.squeeze(corners)
        src_ij = self.get_ij_for_ids(np.squeeze(ids))

        image_size = np.array(image_shape[:2][::-1])

        # find the central corner
        dist = np.linalg.norm(src_xy - image_size/2, axis=1)
        central_corner_idx = np.argmin(dist)

        center_ij = src_ij[central_corner_idx]
        center_xy = src_xy[central_corner_idx]

        square_size_px = 79

        lookup = {ij: xy for ij, xy in zip(src_ij, src_xy)}
        dists = []
        for (i, j), xy0 in lookup.items():
            xy1 = lookup.get((i-1, j))
            if xy1 is not None:
                dists.append(xy0 - xy1)

        dists = np.linalg.norm(np.array(dists), axis=1)

        square_size_px = np.median(dists)
        
        min_ij = np.int32(center_ij - np.ceil(center_xy / square_size_px))
        max_ij = np.int32(center_ij + np.ceil((image_size - center_xy) / square_size_px)) + 1

        min_xy = (min_ij - center_ij) * square_size_px + center_xy
        max_xy = (max_ij - center_ij) * square_size_px + center_xy

        fill_xy, fill_ij = self.fill_missing_corners(src_xy, src_ij, min_ij, max_ij)

        points = ((np.arange(min_ij[0], max_ij[0]) - center_ij[0]) * square_size_px + center_xy[0],
                  (np.arange(min_ij[1], max_ij[1]) - center_ij[1]) * square_size_px + center_xy[1])

        values = np.full((len(points[0]), len(points[1]), 2), np.nan)

        for ij, xy in zip(src_ij, src_xy):
            values[tuple(ij-min_ij)] = xy

        for ij, xy in zip(fill_ij, fill_xy):
            values[tuple(ij-min_ij)] = xy

        interp = scipy.interpolate.RegularGridInterpolator(points, values)

        u_range = np.arange(0, image_shape[1], 1)
        v_range = np.arange(0, image_shape[0], 1)
        u, v = np.meshgrid(u_range, v_range)
        u = u.ravel()
        v = v.ravel()
        remaps = interp((u, v))

        assert len(u) == len(v) == len(remaps) == len(u_range)*len(v_range)

        x_map = np.float32(remaps[:,0].reshape((len(v_range), len(u_range))))
        y_map = np.float32(remaps[:,1].reshape((len(v_range), len(u_range))))

        return x_map, y_map

    def locate_charuco_corners(self, input_image):
        charuco_dict = self.charuco_board.getDictionary()
        
        charuco_params = cv.aruco.CharucoParameters()
        charuco_params.minMarkers = 0
        
        detector = cv.aruco.CharucoDetector(self.charuco_board, charucoParams=charuco_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(input_image)

        return charuco_corners, charuco_ids
        
    def __call__(self, input_image):
        corners, ids = self.locate_charuco_corners(input_image)
        return compute_remaps(corners, ids, input_image.shape)

class CameraThread(threading.Thread):

    def __init__(self, camera, callback):
        super().__init__(daemon=True)
        self.camera = camera
        self.callback = callback

    def run(self):

        while True:
            ret, frame = self.camera.read()
            assert ret
            
            self.callback(frame)

class ScantoolTk:

    def __init__(self, parent, camera_id):

        self._init_charuco_board()
        self._init_camera(camera_id)
        self._init_ui(parent)
        self._init_threads(parent)

    def _init_charuco_board(self):
        chessboard_size = (16, 12)
        square_length = 10.
        marker_length = 5.
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        self.charuco_board = cv.aruco.CharucoBoard(
            chessboard_size, square_length, marker_length, aruco_dict)

    def _init_threads(self, root):
        self.camera_thread = CameraThread(self.camera, self.notify_camera_event)
        root.bind("<<camera>>", self.camera_event)
        self.camera_thread.start()

    def _init_camera(self, camera_id):
        self.camera = None
        
        print("Opening camera...")
        cam = cv.VideoCapture(camera_id, cv.CAP_MSMF)
        assert cam.isOpened()

        w, h = 4656, 3496
        cam.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        w = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

        print("Frame size:", (w,h))
        self.camera = cam
        self.exposure = self.camera.get(cv.CAP_PROP_EXPOSURE)

    def _init_ui(self, parent):
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.canvas_camera = Canvas(self.frame, width=800, height=600,
                                    background='white', highlightthickness=0)
        self.canvas_camera.grid(column=0, row=0, columnspan=2, sticky=(N, W, E, S))

        self.canvas_detail = Canvas(self.frame, width=400, height=300,
                                    background='white', highlightthickness=0)
        self.canvas_detail.grid(column=0, row=1, sticky=(N, W, E, S))

        self.canvas_binary = Canvas(self.frame, width=400, height=300,
                                    background='white', highlightthickness=0)
        self.canvas_binary.grid(column=1, row=1, sticky=(N, W, E, S))

        parent.bind("<Key>", self.key_event)

    def notify_camera_event(self, image):
        self.image_update = image
        self.frame.event_generate("<<camera>>")

    def key_event(self, event):
        if event.char and event.char in '<>':
            self.exposure += 1 if event.char == '>' else -1
            self.camera.set(cv.CAP_PROP_EXPOSURE, self.exposure)
            print(f"exposure={self.exposure} ({event=})")

    def camera_event(self, event):
        image_update = self.image_update
        self.image_update = None

        dst_size = (800, 600)
        image_camera = cv.resize(image_update, dst_size)

        self.update_image_camera(self.maybe_detect_board(image_update))
        self.update_image_detail(image_update)
        self.update_image_binary(image_camera)

    def maybe_detect_board(self, image):
        self.charuco_corners = None
        
        if image.shape[:2] != (3000, 4000):
            return cv.resize(image, (800, 600))

        calibrator = CameraCalibrator(self.charuco_board)
        charuco_corners, charuco_ids = calibrator.locate_charuco_corners(image)

        image = cv.resize(image, (800, 600))
        if charuco_ids is None or len(charuco_ids) == 0:
            return image

        try:
            calibrator.compute_remaps(charuco_corners, charuco_ids, image.shape)
        except Exception as x:
            print(x)
        
        self.draw_detected_corners(image, charuco_corners * .2, charuco_ids)
        
        self.charuco_corners = charuco_corners
        return image

    def update_image_camera(self, image_camera):
        
        self.image_camera = self.to_photo_image(image_camera)
        self.canvas_camera.delete('all')
        self.canvas_camera.create_image((0,0), image=self.image_camera, anchor=NW)

    def update_image_detail(self, image_full):

        src_h, src_w, _ = image_full.shape
        dst_w, dst_h = (400, 300)
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
        
        dst_size = (400, 300)
        self.image_binary = self.to_photo_image(cv.resize(image_binary, dst_size))
        self.canvas_binary.delete('all')
        self.canvas_binary.create_image((0,0), image=self.image_binary, anchor=NW)

    def draw_detected_corners(self, image, corners, ids = None, thickness=1, color=(0,255,255)):
        # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
        shift = 4
        size = 3 << shift
        for pts in np.array(corners * (1 << shift), dtype=np.int32):
            x, y = pts[0]
            cv.rectangle(image, (x-size, y-size), (x+size, y+size), color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
        if ids is not None:
            for pts, id in zip(corners, ids):
                x, y = pts[0]
                cv.putText(image, str(id[0]), (int(x)+5, int(y)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

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
