import csv
import cv2 as cv
import itertools
import numpy as np
import scipy.interpolate
import scipy.spatial.distance
import skimage.draw
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

    def __init__(self, points, poly, verbose=False):

        approx = points[poly]
        # w[i,j] is the straight-line distance from p[i] to p[j]
        self._width = scipy.spatial.distance.cdist(approx, approx)

        def slice(i, j):
            n = len(poly)
            a = poly[i % n]
            b = poly[j % n]
            return points[a:b+1] if a <= b else np.vstack((points[a:], points[:b+1]))

        # l[i] is the distance from p[i-1] to p[i]
        l = np.array([cv.arcLength(slice(i-1, i), closed=False) for i in range(len(poly))])
        if verbose:
            with np.printoptions(precision=3):
                print(f"MagicMatrix:")
                print(f"  {approx=}")
                print(f"  {l=}")
                print(f"  w={self._width}")

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

    def metric(self, ratio):
        return ratio * (self._cumlen - np.atleast_2d(self._cumlen).T) -  self._width

class Dingleberries:

    def __init__(self, epsilon = 5):
        self.epsilon = epsilon
        # maximum width of a dingleberry in pixels
        self.max_width = 24
        # maximum perimeter of a dingleberry in pixels
        self.max_perimeter = 200
        self.cutoff_ratio = 0.5

    @staticmethod
    def to_cuts(candidates):
        if len(candidates) == 0:
            return []
        values = sorted(list(candidates))
        cuts = []
        lo = hi = values[0]
        for v in values[1:]:
            if v == hi+1:
                hi = v
            else:
                cuts.append((lo, hi))
                lo = hi = v
        cuts.append((lo, hi))
        return cuts

    def find_candidate_cuts(self, points):

        points = np.squeeze(points)

        poly = self.get_poly(points)
        mm = MagicMatrix(points, poly)

        candidates = set()
        n = len(poly)
        for i in range(n):
            
            for j in range(i+2,i+n//2):
                width = mm.width(i, j)
                if width > self.max_width:
                    continue

                perimeter = mm.perimeter(i, j)
                if width > self.cutoff_ratio * perimeter:
                    continue

                for k in range(i, j+1):
                    candidates.add(k)

        cuts = self.to_cuts(candidates)
        return [(poly[i], poly[j%len(poly)]) for i, j in cuts]
        
    def optimize_cut(self, points, cut):

        n = len(points)
        return self.refine(points, cut[0] % n, cut[1] % n)

    @staticmethod
    def draw_line(p0, p1):
        # returns an (x,y) tuple and promotes to int64, stack it and
        # take the transpose to get the data format we want
        xy = skimage.draw.line(*p0, *p1)
        return np.array(np.vstack(xy).T, dtype=p0.dtype)
        
    def trim_contour(self, src_points, cuts):

        if len(cuts) == 0:
            return src_points

        ring_slice = puzzler.commands.align.ring_slice

        n = len(src_points)

        # every cut is "unwrapped"
        if not all(i < n and i < j < i+n for i, j in cuts):
            raise ValueError("all cuts should be unwrapped")

        cuts = sorted(cuts)
        
        # the cuts are in order and disjoint
        if not all(a[1] < b[0] for a, b in itertools.pairwise(cuts)):
            raise ValueError("the cuts must be disjoint")

        dst_points = []

        tail = cuts[-1][1]
        
        prev = tail+1 if tail > n else 0
        for i, j in cuts:
            dst_points.append(ring_slice(src_points, prev, i))
            dst_points.append(self.draw_line(src_points[i], src_points[j%n]))
            prev = j + 1
            
        if prev < n:
            dst_points.append(src_points[prev:])

        return np.vstack(dst_points)

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

        metric = mm.metric(0.7)
        
        ii, jj = np.unravel_index(np.argmax(metric), metric.shape)
        return (ii+i, jj+i)
        
    def refine_to_csv(self, points, i, j, verbose=False):

        if verbose:
            print(f"refine: {len(points)=} {i=} {j=}")

        if i <= j:
            poly = np.arange(i, j+1)
        else:
            poly = np.hstack((np.arange(i, len(points)), np.arange(0, j+1)))
        mm = MagicMatrix(points, poly)

        metric = mm.metric(0.7)
        if verbose:
            xx = np.argmax(metric)
            ij = np.unravel_index(xx, metric.shape)
            print(f"max(metric)={ij}+{i}")

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

    def __init__(self, root, args):
        
        puzzle = puzzler.file.load(args.puzzle)
        pieces = dict((i.label, i) for i in puzzle.pieces)

        db = Dingleberries()

        self.pieces = pieces
        self.label = None
        self.points = None
        self.poly = None

        self.candidates = set()

        self.curvature_knots = None
        self.spline = None

        if args.input:
            labels = list()
            with open(args.input, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = row['piece']
                    if p not in labels:
                        labels.append(p)
        else:
            labels = [args.label]

        self.init_ui(root, labels)

        self.select_piece(labels[0])

    def select_piece(self, label):
        
        points = self.pieces[label].points
        
        db = Dingleberries()
        poly = db.get_poly(points)

        print()
        candidates = set()
        for row in db.identify(points):
            i = row['i']
            j = row['j']
            n = len(poly)
            width = row['width']
            perimeter = row['perimeter']
            print(f"{label:4s} {i=:3d} j={j%n:3d} {width=:5.1f} {perimeter=:5.1f}")
            for k in range(i,j+1):
                candidates.add(k)

        def to_ranges(candidates):
            if len(candidates) == 0:
                return []
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
            return ranges

        ranges = to_ranges(candidates)

        ring_slice = puzzler.commands.align.ring_slice

        def fit_spline1(i, j):
            a, b, c, d = [poly[k % len(poly)] for k in (i-1, i, j, j+1)]
            fit_pts = np.vstack((ring_slice(points, a, b), ring_slice(points, c, d)))
            return db.fit_spline(fit_pts)

        def fit_spline2(i, j):
            n = len(poly)
            i, j = poly[i % n], poly[j % n]
            a, b = db.refine(points, i, j)
            print(f"refined: {(i, j)} -> {(a, b)}")
            fit_pts = np.vstack((ring_slice(points, a-12, a), ring_slice(points, b, b+12)))
            return db.fit_spline(fit_pts)

        splines = [fit_spline2(i, j) for i, j in ranges]

        self.label = label
        self.points = points
        self.poly = poly
        self.candidates = candidates
        self.splines = splines
        
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

        for s in self.splines:
            r.draw_lines(s, fill=(0,.5,0,.5), width=4)

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
        
    def change_piece(self, event):
        self.select_piece(self.var_label_combo.get())
        self.render()

    def init_ui(self, parent, labels):

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        f1 = ttk.Frame(parent, padding=5)
        f1.grid_columnconfigure(0, weight=1)
        f1.grid_rowconfigure(0, weight=1)
        f1.grid(column=0, row=0, sticky=(N, W, E, S))

        w, h = 800, 800
        self.canvas = Canvas(f1, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.canvas_wheel)
        self.canvas.bind("<Motion>", self.canvas_motion)
        self.canvas.bind("<Map>", self.canvas_map)

        self.camera = puzzler.commands.align.Camera(np.array((0,0), dtype=np.float64), 1, (w,h))
        self.draggable = None

        f2 = ttk.Frame(f1, padding=5)
        f2.grid(column=1, row=0, sticky=(N, E, S))

        self.var_label_combo = StringVar(value=labels[0])
        cb = ttk.Combobox(f2, textvariable=self.var_label_combo, values=labels, state='readonly')
        cb.grid(column=0, row=0)
        cb.bind("<<ComboboxSelected>>", self.change_piece)

        l = ttk.Label(f2, text='k')
        l.grid(column=0, row=1)

        self.var_k = IntVar(value=10)
        e = ttk.Entry(f2, width=8, textvariable=self.var_k)
        e.grid(column=1, row=1)

        self.var_label = StringVar(value="x,y")
        l = ttk.Label(f2, textvariable=self.var_label, width=40)
        l.grid(column=0, row=2, columnspan=2)

def lint_view(args):

    root = Tk()
    ui = LintTk(root, args)
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

def remove_lint(points):

    db = Dingleberries()

    cuts = db.find_candidate_cuts(points)
    if len(cuts) == 0:
        return points

    optimized_cuts = [db.optimize_cut(points, i) for i in cuts]

    return db.trim_contour(points, optimized_cuts)

def lint_update(args):

    puzzle = puzzler.file.load(args.puzzle)

    for piece in puzzle.pieces:

        piece.points = remove_lint(piece.points)
        piece.tabs = None
        piece.edges = None

    puzzler.file.save(args.puzzle, puzzle)

def add_parser(commands):
    
    parser_lint = commands.add_parser("lint", help="remove lint from outlines")

    commands = parser_lint.add_subparsers()

    parser_view = commands.add_parser("view")
    parser_view.add_argument("label")
    parser_view.add_argument("-i", "--input", help="CSV of pieces to look at")
    parser_view.set_defaults(func=lint_view)

    parser_csv = commands.add_parser("csv")
    parser_csv.add_argument("-o", "--output", required=True)
    parser_csv.set_defaults(func=lint_csv)

    parser_update = commands.add_parser("update", help="remove lint from pieces in the puzzle, updating the puzzle file")
    parser_update.set_defaults(func=lint_update)
