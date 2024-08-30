import argparse
import cv2 as cv
import numpy as np
import PIL.Image
import PIL.ImageTk
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

        self._init_camera(camera_id)
        self._init_ui(parent)
        self._init_threads(parent)

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

    def notify_camera_event(self, image):
        self.image_update = image
        self.frame.event_generate("<<camera>>")

    def camera_event(self, event):
        image = self.image_update
        self.image_update = None

        dst_size = (800, 600)
        image_camera = cv.resize(image, dst_size)
        
        self.image_camera = self.to_photo_image(image_camera)
        self.canvas_camera.delete('all')
        self.canvas_camera.create_image((0,0), image=self.image_camera, anchor=NW)

        src_h, src_w, _ = image.shape
        dst_w, dst_h = (400, 300)
        src_x, src_y = (src_w-dst_w)//2, (src_h-dst_h)//2
        image_detail = image[src_y:src_y+dst_h, src_x:src_x+dst_w]

        self.image_detail = self.to_photo_image(image_detail)
        self.canvas_detail.delete('all')
        self.canvas_detail.create_image((0,0), image=self.image_detail, anchor=NW)

        gray = cv.cvtColor(image_camera, cv.COLOR_BGR2GRAY)
        hist = np.bincount(image_camera.ravel())
        
        thresh = cv.threshold(gray, 84, 255, cv.THRESH_BINARY)[1]
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
            x_coords = np.arange(len(hist))
            y_coords = 280 - (hist * 100) // np.max(hist)
            pts = np.array(np.vstack((x_coords, y_coords)).T, dtype=np.int32)
            # print(f"{x_coords=} {y_coords=} {pts=} {pts.shape=} {pts.dtype=}")
            cv.polylines(image_binary, [pts], False, (255, 255, 0), thickness=2)
            cv.polylines(image_binary, [np.array([(0, 280), (255,280)])], False, (255, 255, 0), thickness=2)
        
        dst_size = (400, 300)
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
