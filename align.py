import argparse
import json
import math
import numpy as np
import PySimpleGUI as sg
import scipy

class AffineTransform:

    def __init__(self):
        self.angle = 0.
        self.dxdy  = np.array((0., 0.))

    def to_global_xy(self, points):
        return points @ self.rot_matrix() + self.dxdy

    def to_local_xy(self, points):
        return (points - self.dxdy) @ self.rot_matrix().T

    def rot_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array(((c, s),
                         (-s, c)))
        
class Piece:

    def __init__(self, path):

        with open(path) as f:
            data = json.load(f)

        contour = np.array(data['contour'], dtype=np.float64)
        self.points = contour - np.mean(contour, axis=0)
        self.coords = AffineTransform()
        self.kdtree = scipy.spatial.KDTree(self.points)

    def dist(self, xy):
        xy = self.coords.to_local_xy(xy)
        return self.kdtree.query(xy, distance_upper_bound=20)

class AlignUI:

    def __init__(self, pieces):
        self.pieces = pieces

        self.drag_active = False
        self.drag_id     = None
        self.drag_start  = None
        self.drag_curr   = None

        self.keep0 = None
        self.keep1 = None

    def find_nearest_piece(self, xy):
        return min((p.dist(xy)[0], i) for i, p in enumerate(self.pieces))

    def graph_drag(self, xy, end):

        if end:

            if self.drag_active and self.drag_id is not None:
                p = self.pieces[self.drag_id]
                dxdy = np.array((self.drag_curr[0]-self.drag_start[0],
                                 self.drag_curr[1]-self.drag_start[1]), dtype=np.float64)
                p.coords.dxdy += dxdy
                
            self.drag_active = False
            self.drag_id     = None
            self.drag_start  = None
            self.drag_curr   = None
            
        elif self.drag_active:

            if self.drag_id is not None:
                self.drag_curr = xy
                
        else:
            self.drag_active = True
            self.drag_start  = xy
            self.drag_curr   = xy

            dist, id = self.find_nearest_piece(xy)
            if dist < 20:
                self.drag_id = id

        self.render()

    def render(self):

        graph = self.window['graph']
        graph.erase()

        colors = ['red', 'green', 'blue']
        for i, p in enumerate(self.pieces):

            color = colors[i%len(colors)]
            dxdy  = p.coords.dxdy
            
            if self.drag_active and i == self.drag_id:
                color = 'cyan'
                dxdy = dxdy + np.array((self.drag_curr[0]-self.drag_start[0],
                                        self.drag_curr[1]-self.drag_start[1]), dtype=np.float64)
                
            points = [tuple(xy + dxdy) for xy in np.squeeze(p.points) @ p.coords.rot_matrix()]
            graph.draw_lines(points, color=color, width=2)

    @staticmethod
    def umeyama(P, Q):
        assert P.shape == Q.shape
        n, dim = P.shape

        centeredP = P - P.mean(axis=0)
        centeredQ = Q - Q.mean(axis=0)

        C = np.dot(np.transpose(centeredP), centeredQ) / n

        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        R = np.dot(V, W)

        varP = np.var(P, axis=0).sum()
        c = 1/varP * np.sum(S) # scale factor

        t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

        return c, R, t

    @staticmethod
    def eldridge(P, Q):
        m = P.shape[0]
        assert P.shape == Q.shape == (m, 2)

        # want the optimized rotation and translation to map P -> Q

        Px, Py = np.sum(P, axis=0)
        Qx, Qy = np.sum(Q, axis=0)

        A00 =np.sum(np.square(P))
        A = np.array([[A00, -Py, Px], 
                      [-Py,  m , 0.],
                      [ Px,  0., m ]], dtype=np.float64)

        u0 = np.sum(P[:,0]*Q[:,1]) - np.sum(P[:,1]*Q[:,0])
        u = np.array([u0, Qx-Px, Qy-Py], dtype=np.float64)

        return np.linalg.lstsq(A, u, rcond=None)

    def do_fit(self):
        
        print("Fit!")

        graph = self.window['graph']

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.points)
        kdtree0 = scipy.spatial.KDTree(points0)
        
        points1 = piece1.coords.to_global_xy(piece1.points)
        kdtree1 = scipy.spatial.KDTree(points1)

        indexes = kdtree0.query_ball_tree(kdtree1, r=15)
        matches = [i for i, v in enumerate(indexes) if len(v)]

        print(f"{len(matches)} points in piece0 have a correspondence with piece1")

        keep = [matches[i] for i in range(0, len(matches), (len(matches)-4) // 4)]
        print(f"{keep=}")

        data0 = points0[keep]
        d, i = kdtree1.query(data0)
        data1 = points1[i]

        for xy in data0:
            graph.draw_point(tuple(xy), color='purple', size=17)

        self.keep0 = keep
        self.keep1 = i

        print(f"{data0=}")
        print(f"{data1=}")

        print(f"umeyama: {self.umeyama(data0, data1)}")
        print(f"eldridge: {self.eldridge(data0, data1)}")

    def do_refit(self):

        if self.keep0 is None or self.keep1 is None:
            return
        
        graph = self.window['graph']

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.points)
        points1 = piece1.coords.to_global_xy(piece1.points)

        coords1 = AffineTransform()
        coords1.angle = piece1.coords.angle + 5 * math.pi / 180.
        coords1.dxdy  = piece1.coords.dxdy
        points1 = coords1.to_global_xy(piece1.points)

        data0 = points0[self.keep0]
        data1 = points1[self.keep1]
        
        for xy in data0:
            graph.draw_point(tuple(xy), color='purple', size=17)
            
        for xy in data1:
            graph.draw_point(tuple(xy), color='purple', size=17)

        print(f"{data0=}")
        print(f"{data1=}")

        print(f"umeyama: {self.umeyama(data0, data1)}")
        print(f"eldridge: {self.eldridge(data0, data1)}")
        
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
                      enable_events=True)],
            [sg.Button('Fit', key='button_fit'),
             sg.Button('Refit', key='button_refit')]
        ]
        
        self.window = sg.Window('Align', layout, finalize=True, return_keyboard_events=True)
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
            elif event == 'button_fit':
                self.do_fit()
            elif event == 'button_refit':
                self.do_refit()
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs='+')

    args = parser.parse_args()

    pieces = [Piece(p) for p in args.pieces]
    
    for i, p in enumerate(pieces):
        p.coords.angle += i * math.pi / 180.

    ui = AlignUI(pieces)
    ui.run()

if __name__ == '__main__':
    main()
