import argparse
import json
import numpy as np
import PySimpleGUI as sg
import scipy

class Piece:

    def __init__(self, path):

        with open(path) as f:
            data = json.load(f)

        self.bbox = tuple(data['bbox'])
        self.contour = np.array(data['contour'], dtype=np.float32)
        self.translation = np.float32([0,0])
        self.kdtree = scipy.spatial.KDTree(self.contour)

    def dist(self, xy):
        xy = np.float32(xy) - self.translation
        return self.kdtree.query(xy, distance_upper_bound=20)

class AlignUI:

    def __init__(self, pieces):
        self.pieces = pieces

        self.drag_active = False
        self.drag_id = None

    def find_nearest_piece(self, xy):
        return min((p.dist(xy)[0], i) for i, p in enumerate(self.pieces))

    def graph_drag(self, xy, end):

        if end:
            self.drag_active = False
            self.drag_id = None
            self.drag_start = None
        elif self.drag_active:

            if self.drag_id is not None:
                piece = self.pieces[self.drag_id]
                dx, dy = xy[0] - self.drag_start[0], xy[1] - self.drag_start[1]
                piece.translation = np.float32((dx,dy))
        else:
            self.drag_active = True
            self.drag_start  = xy

            dist, id = self.find_nearest_piece(xy)
            if dist < 20:
                self.drag_id = id

        self.render()

    def render(self):

        graph = self.window['graph']
        graph.erase()

        colors = ['red', 'green', 'blue']
        for i, p in enumerate(self.pieces):
            color = 'cyan' if i == self.drag_id else colors[i%len(colors)]
            points = [tuple(xy + p.translation) for xy in np.squeeze(p.contour)]
            graph.draw_lines(points, color=color, width=2)

    def _init_ui(self):
        
        w, h = 1024, 1024
        s = 3

        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h * s),
                      graph_top_right = (w * s, 0),
                      background_color='white',
                      key='graph',
                      drag_submits=True,
                      enable_events=True)]
        ]
        
        self.window = sg.Window('Align', layout, finalize=True)
        self.render()

    def run(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == 'graph':
                self.graph_drag(values['graph'], False)
            elif event == 'graph+UP':
                self.graph_drag(values['graph'], True)
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs='+')

    args = parser.parse_args()

    pieces = [Piece(i) for i in args.pieces]

    ui = AlignUI(pieces)
    ui.run()

if __name__ == '__main__':
    main()
