import collections
import cv2 as cv
import math
import numpy as np
import os
import PySimpleGUI as sg
import tempfile
import puzzler

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

class PerimeterUI:

    def __init__(self, puzzle, label):

        piece = None
        for p in puzzle.pieces:
            if p.label == label:
                piece = p
                
        assert piece is not None

        scan = puzzle.scans[piece.source.id]

        scan = puzzler.segment.Segmenter.Scan(scan.path)
        image = scan.get_subimage(piece.source.rect)

        self.pc = PerimeterComputer(image, True)

    def get_image_path(self):

        for label, path in self.pc.images:
            if self.window['radio_image_' + label].get():
                return path
        return None

    def draw_histogram(self):

        graph = self.window['graph']
        
        img = cv.imread(self.get_image_path())
        bgr = len(img.shape) == 3 and img.shape[2]==3

        if bgr:
            for i, color in enumerate(['blue', 'green', 'red']):
                hist = cv.calcHist([img], [i], None, [256], [0, 256])
                scale = 100. / np.max(hist)
                points = [(10+i, 10+scale*v) for i, v in enumerate(np.squeeze(hist))]
                graph.draw_lines(points, color=color)
        else:
            hist = cv.calcHist([img], [0], None, [256], [0, 256])
            scale = 100. / np.max(hist)
            points = [(10+i, 10+scale*v) for i, v in enumerate(np.squeeze(hist))]
            graph.draw_lines(points, color='black')

    def draw_contour(self):
        graph = self.window['graph']
        
        points = [tuple(xy) for xy in np.squeeze(self.pc.contour)]
        graph.draw_lines(points, color='red', width=2)

    def render(self):

        graph = self.window['graph']
        graph.erase()

        path = self.get_image_path()
        graph.draw_image(filename=path, location=(0,self.pc.image_size[1]))

        if self.window['render_histogram'].get():
            self.draw_histogram()

        if self.window['render_contour'].get():
            self.draw_contour()

    def _init_ui(self):
        
        w, h = self.pc.image_size

        controls = []
        
        for image in self.pc.images:
            label = image[0]
            is_default = 0 == len(controls)
            key = 'radio_image_' + label
            r = sg.Radio(label, 'radio_image', default=is_default, enable_events=True, key=key)
            controls.append(r)

        controls.append(sg.CB('Render Histogram', default=False, enable_events=True, key='render_histogram'))
        controls.append(sg.CB('Render Contour', default=True, enable_events=True, key='render_contour'))
        
        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, 0),
                      graph_top_right = (w, h),
                      background_color='black',
                      key='graph',
                      enable_events=True),
             controls],
        ]
        
        self.window = sg.Window('Perimeter Computer', layout, finalize=True)
        self.render()

    def run(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event.startswith('radio_image_'):
                self.render()
            elif event.startswith('render_'):
                self.render()
            else:
                print(event, values)


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
    ui = PerimeterUI(puzzle, args.label)
    ui.run()
    
def add_parser(commands):
    parser_points = commands.add_parser("points", help="outline pieces")

    commands = parser_points.add_subparsers()

    parser_update = commands.add_parser("update", help="update the point computation in puzzle")
    parser_update.set_defaults(func=points_update)

    parser_view = commands.add_parser("view", help="view the points computation for a specific image")
    parser_view.add_argument("label")
    parser_view.set_defaults(func=points_view)
