import argparse
import json
import numpy as np
import PySimpleGUI as sg
import scipy

class Piece:

    def __init__(self, path, rotate_degrees=None):

        with open(path) as f:
            data = json.load(f)

        self.bbox = tuple(data['bbox'])
        self.contour = np.array(data['contour'], dtype=np.float32)
        
        if rotate_degrees is not None and rotate_degrees != 0.:
            print(f"{rotate_degrees=}")
            print(f"{self.contour=}")
            rad = np.radians(rotate_degrees)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array(((c, -s), (s, c)))
            print(f"{rot=}")
            self.contour = self.contour @ rot
            print(f"{self.contour=}")
            
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

        points0 = self.pieces[0].contour
        kdtree0 = self.pieces[0].kdtree
        
        points1 = self.pieces[1].contour + self.pieces[1].translation
        kdtree1 = scipy.spatial.KDTree(points1)

        print(f"{self.pieces[1].contour=}")
        print(f"{self.pieces[1].translation=}")
        print(f"{points1=}")
 
        indexes = kdtree0.query_ball_tree(kdtree1, r=15)
        matches = [i for i, v in enumerate(indexes) if len(v)]

        print(f"{len(matches)} points in piece0 have a correspondence with piece1")

        keep = [matches[i] for i in range(0, len(matches), (len(matches)-4) // 4)]
        print(f"{keep=}")

        for i in keep:
            xy = tuple(points0[i])
            graph.draw_point(xy, color='purple', size=17)

        data0 = self.pieces[0].contour[keep]
        d, i = kdtree1.query(data0)
        data1 = points1[i]

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
            [sg.Button('Fit', key='button_fit')]
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
            elif event == 'button_fit':
                self.do_fit()
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs='+')

    args = parser.parse_args()

    pieces = [Piece(p, rotate_degrees=i*5) for i,p in enumerate(args.pieces)]

    ui = AlignUI(pieces)
    ui.run()

if __name__ == '__main__':
    main()
