import cv2 as cv
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

    def __init__(self, parent):

        self._init_camera()
        self._init_ui(parent)
        self._init_threads(parent)

    def _init_threads(self, root):
        self.camera_thread = CameraThread(self.camera, self.notify_camera_event)
        root.bind("<<camera>>", self.camera_event)
        self.camera_thread.start()

    def _init_camera(self):
        self.camera = None
        
        print("Opening camera...")
        cam = cv.VideoCapture(2, cv.CAP_MSMF)
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
        self.canvas_camera.grid(column=0, row=0, sticky=(N, W, E, S))

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
        
        image_camera = PIL.Image.frombuffer('RGB', dst_size, image_camera.tobytes(), 'raw', 'BGR', 0, 1)
        self.image_camera = PIL.ImageTk.PhotoImage(image=image_camera)
        self.canvas_camera.delete('all')
        self.canvas_camera.create_image((0,0), image=self.image_camera, anchor=NW)

        src_h, src_w, _ = image.shape
        dst_w, dst_h = (400, 300)
        src_x, src_y = (src_w-dst_w)//2, (src_h-dst_h)//2
        image_detail = image[src_y:src_y+dst_h, src_x:src_x+dst_w]

        # print(f"{image.shape=} {image_detail.shape=}")

        image_detail = PIL.Image.frombuffer('RGB', (dst_w, dst_h), image_detail.tobytes(), 'raw', 'BGR', 0, 1)
        self.image_detail = PIL.ImageTk.PhotoImage(image=image_detail)
        self.canvas_detail.delete('all')
        self.canvas_detail.create_image((0,0), image=self.image_detail, anchor=NW)

def main():
    root = Tk()
    ui = ScantoolTk(root)
    root.title("PuZzLeR: scantool")
    root.mainloop()

if __name__ == '__main__':
    main()
