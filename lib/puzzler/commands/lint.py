import csv
import cv2 as cv
import numpy as np
import scipy.interpolate
import scipy.spatial.distance
import puzzler
from tqdm import tqdm

from tkinter import *
from tkinter import ttk

class Curvature:

    def __init__(self, k, points):
        self.k = k
        self.points = np.squeeze(points)

    def knots_for_point(self, i):

        k, n = self.k, len(self.points)
        
        p0 = self.points[(i-k)%n]
        p1 = self.points[i]
        p2 = self.points[(i+k)%n]

        return (p0, p1, p2)

    @staticmethod
    def equation_for_knots(knots):

        p0, p1, p2 = knots
        
        c0 = p1
        c1 = (p2 - p0) * .5
        c2 = (p2 + p0) * .5 - p1

        return (c0, c1, c2)

    def equation_for_point(self, i):

        return self.equation_for_knots(self.knots_for_point(i))

    def curvature_at_point(self, i):

        c0, c1, c2 = self.equation_for_point(i)

        a1, b1 = c1
        a2, b2 = c2

        return 2 * (a1*b2 - a2*b1) / ((a1*a1 + b1*b1) ** 1.5)

class MagicMatrix:

    def __init__(self, points, poly):

        approx = points[poly]
        # w[i,j] is the straight-line distance from p[i] to p[j]
        self._width = scipy.spatial.distance.cdist(approx, approx)

        def slice(i, j):
            a = poly[i-1]
            b = poly[i]
            return points[a:b+1] if a <= b else np.vstack((points[a:], points[:b+1]))

        # l[i] is the distance from p[i-1] to p[i]
        l = [cv.arcLength(slice(i-1, i), closed=False) for i in range(len(poly))]

        # cl[j] - cl[i] is the total path length from p[i] to p[j]
        self._cumlen = np.cumsum(l)

    def width(self, i, j):
        n = self._width.shape[0]
        return self._width[i%n, j%n]

    def perimeter(self, i, j):
        n = self._cumlen.shape[0]
        i %= n
        j %= n
        p = self._cumlen[j] - self._cumlen[i]
        if j < i:
            p += self._cumlen[-1]
        return p

class Dingleberries:

    def __init__(self, epsilon = 5):
        self.epsilon = epsilon
        # maximum width of a dingleberry in pixels
        self.max_width = 24
        # maximum perimeter of a dingleberry in pixels
        self.max_perimeter = 200
        self.cutoff_ratio = 0.5

    def get_poly(self, points):
        # HACK: prevent sample points that are too close where we wrap around
        return np.arange(6, len(points)-5, 12)

    def identify(self, points):

        points = np.squeeze(points)

        poly = self.get_poly(points)
        mm = MagicMatrix(points, poly)

        retval = []
        n = len(poly)
        for i in range(n):
            
            for j in range(i+2,i+n//2):
                width = mm.width(i, j)
                if width > self.max_width:
                    continue

                perimeter = mm.perimeter(i, j)
                if width > self.cutoff_ratio * perimeter:
                    continue

                retval.append({'i':i, 'j':j, 'pi':poly[i], 'pj':poly[j%n], 'width':width, 'perimeter':perimeter})

        return retval

    def refine(self, points, i, j):

        if i <= j:
            poly = np.arange(i, j+1)
        else:
            poly = np.hstack((np.arange(i, len(points)), np.arange(0, j+1)))
        mm = MagicMatrix(points, poly)

        # print(f"{mm._cumlen=}")

        metric = 0.7 * (mm._cumlen - np.atleast_2d(mm._cumlen).T) -  mm._width

        f = open('refine_lint.csv', 'w', newline='')
        writer = csv.DictWriter(f, fieldnames='i j width perimeter metric'.split())
        writer.writeheader()

        n = len(poly)
        for a in range(1,n):
            for b in range(a+1,n):
                row = {'i':a+i, 'j':b+i, 'width':mm.width(a,b), 'perimeter':mm.perimeter(a,b), 'metric':metric[a,b]}
                for k in 'width', 'perimeter', 'metric':
                    row[k] = f"{row[k]:.3f}"
                writer.writerow(row)

        f.close()

    def fit_spline(self, points):

        assert isinstance(points, np.ndarray)
        x, y = points[:,0], points[:,1]
        tck, u = scipy.interpolate.splprep([x, y], s=len(x)*.1)
        out = scipy.interpolate.splev(np.linspace(0,1,20), tck)
        return np.vstack((out[0], out[1])).T

class LintTk:

    def __init__(self, root, pieces, label):
        
        db = Dingleberries()

        self.pieces = pieces
        self.label = label
        self.points = self.pieces[label].points
        self.poly = db.get_poly(self.points)

        self.curvature_knots = None
        self.spline = None

        self.init_ui(root)

        self.candidates = set()
        for row in db.identify(self.points):
            i = row['i']
            j = row['j']
            n = len(self.poly)
            width = row['width']
            perimeter = row['perimeter']
            print(f"{i=:3d} {j=:3d} {width=:5.1f} {perimeter=:5.1f}")
            for k in range(i,j+1):
                self.candidates.add(k % n)

        sp_dict = {
            'F24': (48, 49, 52, 53),
            'F24x': (201, 202, 1, 2),
            'F18': (71, 72, 84, 85),
            'U26': (86, 87, 92, 93),
            'A1': (189, 190, 192, 193),
            'A2': (93, 94, 99, 100),
            'A12': (60, 61, 65, 66),
            'B24': (85, 86, 93, 94),
            'B24x': (173, 174, 178, 179),
            'E25': (117, 118, 124, 125),
            'K33': (65, 66, 70, 71),
            'L28': (22, 23, 25, 26),
            'M20': (24, 25, 28, 29),
        }

        if label in sp_dict:
            n = len(self.poly)
            sp = sp_dict[label]
            if len(sp) == 4:
                a, b, c, d = [self.poly[i%n] for i in sp]
                fit_pts = np.vstack((self.points[a:b], self.points[c:d]))
            else:
                fit_pts = np.array([self.points[self.poly[i%n]] for i in sp])
            self.spline = db.fit_spline(fit_pts)
            # print(f"{fit_pts=} {self.spline=}")

            sp = sp_dict[label]
            db.refine(self.points, self.poly[sp[0]%n], self.poly[sp[-1]%n])

    def render(self):

        r = puzzler.renderer.cairo.CairoRenderer(self.canvas)

        r.transform(self.camera.matrix)
        
        r.draw_points(self.points, radius=1, fill='black')

        if self.curvature_knots is not None:
            r.draw_points(self.curvature_knots, radius=4, fill='blue')

            t = np.linspace(-3, 3, 50)

            eqn = Curvature.equation_for_knots(self.curvature_knots)
            pts = np.atleast_2d(eqn[0]).T + np.atleast_2d(eqn[1]).T * t + np.atleast_2d(eqn[2]).T * np.square(t)

            r.draw_points(pts.T, radius=2, fill='green')

        if self.poly is not None:
            f = r.make_font('Courier New', 12)
            for i, j in enumerate(self.poly):
                xy = self.points[j]
                color = 'red' if i in self.candidates else 'blue'
                r.draw_points([xy], radius=10, fill=color, outline='')
                r.draw_text(xy, text=f"{i}", font=f, fill='white')

        if self.spline is not None:
            r.draw_lines(self.spline, fill=(0,.5,0,.5), width=8)

        self.displayed_image = r.commit()

    def canvas_map(self, event):
        self.render()

    def canvas_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.canvas_motion(event)
        self.render()

    def canvas_press(self, event):

        self.draggable = puzzler.commands.align.MoveCamera(self.camera)
        self.draggable.start(np.array((event.x, event.y)))

        self.render()

    def canvas_drag(self, event):

        if not self.draggable:
            return
        
        self.draggable.drag(np.array((event.x, event.y)))
        
        self.render()

    def canvas_release(self, event):

        if not self.draggable:
            return
        
        self.draggable.commit()
        self.draggable = None

        self.render()

    def canvas_device_to_user(self, x, y):
        cm = self.camera.matrix
        xy0 = np.array((x, y, 1))
        xy1 = xy0 @ np.linalg.inv(cm).T
        return xy1[:2]
        
    def canvas_motion(self, event):
        xy = self.canvas_device_to_user(event.x, event.y)
        d2 = np.sum(np.square(self.points - xy), axis=1)
        i = np.argmin(d2)

        s = f"{xy[0]:.0f},{xy[1]:.0f}"
        if d2[i] < 1000:
            s += f": point={i}"
            # self.curvature_knots = Curvature(self.var_k.get(), self.points).knots_for_point(i)
            # self.render()

        self.var_label.set(s)
        
    def init_ui(self, parent):

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        f1 = ttk.Frame(parent, padding=5)
        f1.grid(column=0, row=0, sticky=(N, W, E, S))
        f1.grid_rowconfigure(0, weight=1)
        f1.grid_columnconfigure(0, weight=1)
        
        f2 = ttk.Frame(f1, padding=0)
        f2.grid_columnconfigure(0, weight=1)
        f2.grid_rowconfigure(0, weight=1)
        f2.grid(column=0, row=0, sticky=(N, W, E, S))

        w, h = 800, 800
        self.canvas = Canvas(f2, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.canvas_wheel)
        self.canvas.bind("<Motion>", self.canvas_motion)
        self.canvas.bind("<Map>", self.canvas_map)

        self.camera = puzzler.commands.align.Camera(np.array((0,0), dtype=np.float64), 1, (w,h))
        self.draggable = None

        f3 = ttk.Frame(f2, padding=5)
        f3.grid(column=1, row=0, sticky=(N, E, S))

        l = ttk.Label(f3, text='k')
        l.grid(column=0, row=0)

        self.var_k = IntVar(value=10)
        e = ttk.Entry(f3, width=8, textvariable=self.var_k)
        e.grid(column=1, row=0)

        self.var_label = StringVar(value="x,y")
        l = ttk.Label(f3, textvariable=self.var_label, width=40)
        l.grid(column=0, row=1, columnspan=2)

        self.trace = Canvas(f1, width=w, height=100, background='grey', highlightthickness=0)
        self.trace.grid(column=0, row=1, sticky=(W, E))

def lint_view(args):

    puzzle = puzzler.file.load(args.puzzle)
    pieces = dict((i.label, i) for i in puzzle.pieces)

    root = Tk()
    ui = LintTk(root, pieces, args.label)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: lint view")
    root.mainloop()

def lint_csv(args):

    def format_as_ranges(candidates):
        values = sorted(list(candidates))
        ranges = []
        lo = hi = values[0]
        for v in values[1:]:
            if v == hi+1:
                hi = v
            else:
                ranges.append((lo, hi))
                lo = hi = v
        ranges.append((lo, hi))

        ret = []
        for lo, hi in ranges:
            if lo == hi:
                ret.append(str(lo))
            else:
                ret.append(str(lo) + '-' + str(hi))
        return ', '.join(ret)
            
    def process_rows(label, rows):

        candidates = set()
        for row in rows:
            row['piece'] = label
            for key in 'width', 'perimeter':
                row[key] = f"{row[key]:.1f}"
            for k in range(row['i'], row['j']+1):
                candidates.add(k)

        if len(candidates):
            s = format_as_ranges(candidates)
            print(f"{label:4s}: {s}")

        return rows
                
    puzzle = puzzler.file.load(args.puzzle)
    
    db = Dingleberries()

    with open(args.output, 'w', newline='') as f:
        
        writer = csv.DictWriter(f, fieldnames='piece i j pi pj width perimeter'.split())
        writer.writeheader()
        
        for piece in puzzle.pieces:
            try:
                writer.writerows(process_rows(piece.label, db.identify(piece.points)))
            except ValueError as x:
                print(f"problem with piece={piece.label}")
                print(x)
                
def add_parser(commands):
    
    parser_lint = commands.add_parser("lint", help="remove lint from outlines")

    commands = parser_lint.add_subparsers()

    parser_view = commands.add_parser("view")
    parser_view.add_argument("label")
    parser_view.set_defaults(func=lint_view)

    parser_csv = commands.add_parser("csv")
    parser_csv.add_argument("-o", "--output", required=True)
    parser_csv.set_defaults(func=lint_csv)
