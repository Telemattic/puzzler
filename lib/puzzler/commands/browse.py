import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import puzzler

class Browser:

    class Outline:

        def __init__(self, piece, epsilon=10):

            points = np.array(puzzler.chain.ChainCode().decode(piece['points']))
            approx  = cv.approxPolyDP(points, epsilon, True)
            self.poly = np.squeeze(approx)

            ll = np.min(self.poly, 0)
            ur = np.max(self.poly, 0)
            self.bbox   = tuple(ll.tolist() + ur.tolist())
            self.label  = piece['label']

            # print(f"{self.poly=} {self.bbox=} {self.label=}")

    def __init__(self, puzzle):

        outlines = [Browser.Outline(p) for p in puzzle['pieces']]

        cols = 1
        while cols * cols < len(outlines):
            cols += 1

        rows = 1
        while cols * rows < len(outlines):
            rows += 1

        self.outlines = outlines
        self.cols = cols
        self.rows = rows

        max_w = 1400
        max_h = 800

        bbox_w = max(o.bbox[2] - o.bbox[0] for o in self.outlines)
        bbox_h = max(o.bbox[3] - o.bbox[1] for o in self.outlines)

        self.scale = min(max_w / (cols * bbox_w), max_h / (rows * bbox_h))
        self.tile_w = max_w // cols
        self.tile_h = max_h // cols
        self.width  = self.tile_w * self.cols
        self.height = self.tile_h * self.cols

    def render(self, graph):

        for i, o in enumerate(self.outlines):
            x = (i %  self.cols) * self.tile_w + self.tile_w // 2
            y = (i // self.cols) * self.tile_h + self.tile_h // 2

            # want the corners of the outline bbox centered within the tile
            bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2
            points = (o.poly - bbox_center) * self.scale + np.array((x, y))

            graph.draw_lines(points, color='black')

class BrowseUI:

    def __init__(self, puzzle):

        self.browser = Browser(puzzle)

    def render(self):

        graph = self.window['graph']
        graph.erase()

        self.browser.render(graph)

    def _init_ui(self):
        
        w, h = self.browser.width, self.browser.height

        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h),
                      graph_top_right = (w, 0),
                      background_color='white',
                      key='graph',
                      enable_events=True)]
        ]
        
        self.window = sg.Window('Browser', layout, finalize=True)
        self.render()

    def run(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            else:
                print(event, values)

def browse(args):

    puzzle = puzzler.file.load(args.puzzle)
    ui = BrowseUI(puzzle)
    ui.run()

def add_parser(commands):
    parser_browse = commands.add_parser("browse", help="browse pieces")
    parser_browse.set_defaults(func=browse)
