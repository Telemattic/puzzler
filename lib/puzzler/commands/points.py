import collections
import cv2 as cv
import math
import numpy as np
import os
import tempfile
import puzzler

from tkinter import *
from tkinter import ttk

class PerimeterComputer:

    def __init__(self, img, save_images = False):
        
        self.tempdir = None
        if save_images:
            self.tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp')

        assert img is not None
        
        w, h = img.shape[1], img.shape[0]
        self.image_size  = (w,h)
        self.images = []

        print(f"{w}x{h}")
        
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur0    = cv.medianBlur(gray, 7)
        blur1    = cv.medianBlur(gray, 31)

        xweight  = np.sin(np.linspace(0, math.pi, num=w))
        yweight  = np.sin(np.linspace(0, math.pi, num=h))
        weight   = yweight[:,np.newaxis] @ xweight[np.newaxis,:]

        blur     = np.uint8(blur0 * (1. - weight) + blur1 * weight)

        weight   = np.uint8(weight * 255 / np.max(weight))
        self._add_temp_image("weight.png", weight, "Weight")
        
        thresh   = cv.threshold(blur, 107, 255, cv.THRESH_BINARY)[1]

        self._add_temp_image("color.png", img, 'Source')
        self._add_temp_image("gray.png", gray, 'Gray')
        self._add_temp_image("blur0.png", blur0, 'Blur 0')
        self._add_temp_image("blur1.png", blur1, 'Blur 1')
        self._add_temp_image("blur.png", blur, 'Blur')
        self._add_temp_image("thresh.png", thresh, 'Thresh')
        
        contours = cv.findContours(np.flip(thresh, axis=0), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        self.contour = max(contours[0], key=cv.contourArea)

    def _add_temp_image(self, filename, img, label):

        if self.tempdir is None:
            return
        
        path = os.path.join(self.tempdir.name, filename)
        cv.imwrite(path, img)
        self.images.append((label, path))

class PerimeterTk:

    def __init__(self, parent, puzzle, label):
        piece = None
        for p in puzzle.pieces:
            if p.label == label:
                piece = p
                
        assert piece is not None

        scan = puzzle.scans[piece.source.id]

        scan = puzzler.segment.Segmenter.Scan(scan.path)
        image = scan.get_subimage(piece.source.rect)

        self.pc = PerimeterComputer(image, True)
        w, h = self.pc.image_size

        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(column=1, row=0, sticky=(N, W, E, S))

        self.label = ttk.Label(self.controls, text=f"{label} {w}x{h}")
        self.label.grid(column=0, row=0, sticky=(N, W), padx=10)
        
        self.image_names = StringVar(value=[label for label, _ in self.pc.images])
        self.lbox = Listbox(self.controls, listvariable=self.image_names, height=len(self.pc.images))
        self.lbox.grid(column=0, row=1, sticky=(N, W), padx=10)
        self.lbox.bind('<<ListboxSelect>>', self.render)
        self.lbox.selection_set(0)

        self.check_var = IntVar(value=1)
        self.check = ttk.Checkbutton(self.controls, text='Render Contour', command=self.render,
                                     variable=self.check_var)
        self.check.grid(column=0, row=2, sticky=(N, W), padx=10)

        self.canvas = Canvas(self.frame, width=w, height=h, background='blue', highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))

        self.render()

    def render(self, *args):

        self.canvas.delete('all')
        image = PhotoImage(file=self.get_image_path())
        self.canvas.create_image((0, 0), image=image, anchor=NW)
        self.displayed_image = image

        if self.check_var.get():
            points = self.pc.contour * np.array((1, -1)) + np.array((0, image.height()-1))
            self.canvas.create_polygon(points.tolist(), outline='red', fill='')

    def get_image_path(self):
        i = self.lbox.curselection()[0]
        return self.pc.images[i][1]

def points_update(args):

    puzzle = puzzler.file.load(args.puzzle)

    pieces_by_source = collections.defaultdict(list)
    for p in puzzle.pieces:
        s = p.source.id
        pieces_by_source[s].append(p)

    for source_id, pieces in pieces_by_source.items():

        scan = puzzle.scans[source_id]

        print(scan.path)

        scan = puzzler.segment.Segmenter.Scan(scan.path)
        for piece in pieces:
            image = scan.get_subimage(piece.source.rect)
            pc = PerimeterComputer(image)
            piece.points = np.squeeze(pc.contour)
            piece.tabs = None
            piece.edges = None
    
    puzzler.file.save(args.puzzle, puzzle)

def points_view(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    root = Tk()
    ui = PerimeterTk(root, puzzle, args.label)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: points")
    root.wm_resizable(0, 0)
    root.mainloop()
    
def add_parser(commands):
    parser_points = commands.add_parser("points", help="outline pieces")

    commands = parser_points.add_subparsers()

    parser_update = commands.add_parser("update", help="update the point computation in puzzle")
    parser_update.set_defaults(func=points_update)

    parser_view = commands.add_parser("view", help="view the points computation for a specific image")
    parser_view.add_argument("label")
    parser_view.set_defaults(func=points_view)
