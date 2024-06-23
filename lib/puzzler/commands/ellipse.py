import bisect
import cv2 as cv
import itertools
import math
import numpy as np
import operator
import puzzler
import puzzler.renderer.canvas

from tkinter import *
from tkinter import ttk

from tqdm import tqdm

class PerimeterLoader:

    def __init__(self, piece):
        points = piece.points
        assert points.ndim == 2 and points.shape[1] == 2
        ll = np.min(points, 0)
        ur = np.max(points, 0)
        # print(f"{points=} {ll=} {ur=}")
        self.bbox   = tuple(ll.tolist() + ur.tolist())
        self.points = points
        # print(f"{self.bbox=} {self.points.shape=}")
        self.index = dict((tuple(xy), i) for i, xy in enumerate(points))

    def slice(self, a, b):
        return np.vstack((self.points[a:] , self.points[:b])) if a > b else self.points[a:b]

class ApproxPolyComputer:

    def __init__(self, perimeter, epsilon):
        self.epsilon = epsilon
        self.indexes = self._compute_poly(perimeter, epsilon)
        self.signed_area = self._compute_signed_areas(perimeter, self.indexes)

    def curvature_runs(self):

        indexes, signed_area = self.indexes, self.signed_area
        signs = [(area > 0, i) for i, area in zip(indexes, signed_area)]
        return [(k,list(i for _, i in g)) for k, g in itertools.groupby(signs, key=operator.itemgetter(0))]
    
    @staticmethod
    def _compute_poly(perimeter, epsilon):
        
        approx = cv.approxPolyDP(perimeter.points, epsilon, True)
        poly = list(perimeter.index[tuple(xy)] for xy in np.squeeze(approx))

        reset = None
        for i in range(1,len(poly)):
            if poly[i-1] > poly[i]:
                assert reset is None
                reset = i
            else:
                assert poly[i-1] < poly[i]
                
        if reset:
            poly = poly[reset:] + poly[:reset]

        return poly

    @staticmethod
    def _compute_signed_areas(perimeter, indexes):
        points = np.squeeze(perimeter.points)
        n = len(indexes)
        signed_areas = []
        for i in range(n):
            if i+1 >= n:
                i -= n
            x0, y0 = points[indexes[i-1]]
            x1, y1 = points[indexes[i]]
            x2, y2 = points[indexes[i+1]]
            area = (x1-x0) * (y2-y1) - (x2-x1)*(y1-y0)
            signed_areas.append(area)
        return signed_areas

class TabComputer:

    def __init__(self, perimeter, epsilon, verbose=True):
        
        self.perimeter = perimeter
        self.verbose   = verbose
        self.approx_poly = ApproxPolyComputer(self.perimeter, epsilon)
        self.max_mse   = 500
        self.min_angle = math.radians(220)

        self.tabs = []

        for defect in self.compute_convexity_defects(self.perimeter, self.approx_poly):

            if defect[3] < 8000:
                continue
            
            l, r, c = defect[0], defect[1], defect[2]
            tab = self.fit_ellipse_to_convexity_defect(l, r, c)
            if tab is None:
                continue

            if tab['angle'] < self.min_angle:
                if self.verbose:
                    print("  angle for ellipse too small, rejecting")
                continue

            if tab['mse'] > self.max_mse:
                if self.verbose:
                    print("  MSE of fit too high, rejecting")
                continue
            
            e = tab['ellipse']

            if e.semi_major / e.semi_minor > 1.8:
                if self.verbose:
                    print("  ellipse too eccentric, rejecting")
                continue

            if self.verbose:
                print("  ** keeping indent tab")
            
            self.tabs.append(tab)

        for in_out, indices in self.approx_poly.curvature_runs():

            if in_out:
                continue
            
            tab = self.fit_ellipse_to_outdent(indices[0], indices[-1])
            if tab is None:
                continue

            if tab['angle'] < self.min_angle:
                if self.verbose:
                    print("  angle for ellipse too small, rejecting")
                continue
            
            if tab['mse'] > self.max_mse:
                if self.verbose:
                    print("  MSE of fit too high, rejecting")
                continue
            
            e = tab['ellipse']

            if e.semi_major / e.semi_minor > 1.8:
                if self.verbose:
                    print("  ellipse too eccentric, rejecting")
                continue

            if e.semi_major > 90:
                if self.verbose:
                    print("   ellipse too big, rejecting")
                continue

            if self.indexes_overlap(tab):
                if self.verbose:
                    print("  outdent tab overlaps previously found tab")
                continue
            
            if self.verbose:
                print("  ** keeping outdent tab")
            
            self.tabs.append(tab)

        return

        if self.verbose:
            print(f"Iterate fit process for {len(self.tabs)} ellipses")

        for j in range(3):
            if self.verbose:
                print(f"-- pass {j+1}/3 --")
            for i, tab in enumerate(self.tabs):
                a, b = tab['trimmed_indexes']
                if self.verbose:
                    print(f"{i=}: indexes={tab['indexes']} trimmed={tab['trimmed_indexes']}")
                tab2 = self.fit_ellipse(a, b, tab['indent'])
                if tab2 is None:
                    print("FNORD!")
                else:
                    self.tabs[i] = tab2

    def indexes_overlap(self, tab):
        
        n = len(self.perimeter.points)

        def overlaps(i, j):

            a0, b0 = i
            if b0 < a0:
                return overlaps((a0, n), j) or overlaps((0, b0), j)
            
            a1, b1 = j
            if b1 < a1:
                return overlaps(i, (a1, n)) or overlaps(i, (0, b1))

            return a1 < b0 and a0 < b1
        
        for other in self.tabs:

            if overlaps(tab['indexes'], other['indexes']):
                return True

        return False
            
    def fit_ellipse_to_outdent(self, l, r):

        if self.verbose:
            print(f"\nfit_ellipse_to_outdent: {l=} {r=}")

        indexes, signed_area = self.approx_poly.indexes, self.approx_poly.signed_area

        aa = a = bisect.bisect_left(indexes, l)
        while signed_area[a] < 0:
            a -= 1

        bb = b = bisect.bisect_left(indexes, r)
        n = len(signed_area)
        while signed_area[b] < 0:
            b += 1
            if b == n:
                return None

        if self.verbose:
            print(f"  approx: {aa=} {a=} {bb=} {b=} {indexes[a]=} {indexes[b]=}")

        return self.fit_ellipse(indexes[a], indexes[b], False)

    def fit_ellipse_to_convexity_defect(self, l, r, c):

        if self.verbose:
            print(f"\nfit_ellipse_to_convexity_defect: {l=} {r=} {c=}")

        indexes, signed_area = self.approx_poly.indexes, self.approx_poly.signed_area
        n = len(indexes)

        assert 0 <= l < n and 0 <= r < n

        a = c
        while a != l and signed_area[a] > 0:
            a -= 1
            if a < 0:
                a = n-1

        b = c
        while b != r and signed_area[b] > 0:
            b += 1
            if b == n:
                b = 0

        if self.verbose:
            print(f"  approx: {a=} {b=}")

        return self.fit_ellipse(indexes[a], indexes[b], True)

    def fit_ellipse(self, a, b, indent):

        if self.verbose:
            print(f"fit_ellipse: {a=} {b=} {indent=}")

        points = self.perimeter.slice(a, b)
        ellipse = puzzler.geometry.fit_ellipse_to_points(points)
        if ellipse is None:
            return None

        dist = puzzler.geometry.DistanceToEllipseComputer(ellipse)(points)
        sse  = np.sum(dist ** 2)
        mse  = sse / len(points)
            
        if self.verbose:
            print(f"  fit: SSE={sse:.1f} MSE={mse:.1f}")

        poly = [ellipse.center[0], ellipse.center[1], ellipse.semi_major, ellipse.semi_minor, None, ellipse.phi]

        center = np.array(poly[:2])
        cx, cy = center[0], center[1]

        if indent:
            angle = self.angle_between(center, b, a)
        else:
            angle = self.angle_between(center, a, b)

        if self.verbose:
            print(f"  center={cx:.1f},{cy:.1f} major={ellipse.semi_major:.1f} minor={ellipse.semi_minor:.1f} phi={math.degrees(ellipse.phi):.1f}")
            print(f"  angle={math.degrees(angle):.1f}")

        tab = {'ellipse': ellipse, 'indexes': (a,b), 'indent':indent, 'angle': angle, 'mse': mse}

        # indices of last two points that are "close enough" to the ellipse
        ## aa, bb = self.find_fit_range(ellipse)

        # indices of tangent points
        self.find_tangent_points(tab)

        self.trim_indexes(tab)

        if self.verbose:
            print(f"  indexes: fit={tab['indexes']} tangent={tab['tangents']} trimmed={tab['trimmed_indexes']}")

        return tab

    def angle_between(self, center, a, b):

        n = len(self.perimeter.points)
        pa = self.perimeter.points[a%n] - center
        pb = self.perimeter.points[b%n] - center
        pa = pa / np.linalg.norm(pa)
        pb = pb / np.linalg.norm(pb)
        dot = np.sum(pa * pb)
        angle = np.arccos(dot)
        if np.cross(pa, pb) > 0:
            angle = math.pi * 2 - angle
        return angle

    def trim_indexes(self, tab):

        dist = puzzler.geometry.DistanceToEllipseComputer(tab['ellipse'])

        a, b = tab['indexes']
        points = self.perimeter.points 
        n = len(points)

        va = []
        for aa in range(a,a+50):
            d = dist(points[aa % n])
            va.append(d)
            if d < 5:
                break

        vb = []
        for bb in range(b,b-50,-1):
            d = dist(points[bb])
            vb.append(d)
            if d < 5:
                break

        if False and self.verbose:
            print("trim_indexes:")
            with np.printoptions(precision=1):
                print(f"  {a=} {aa=} {np.array(va)}")
                print(f"  {b=} {bb=} {np.array(vb)}")

        tab['trimmed_indexes'] = (aa, bb)

    def find_tangent_points(self, tab):

        center = tab['ellipse'].center

        a, b = tab['indexes']
        # print(f"find_tangent_points: center=({center[0]:.1f},{center[1]:.1f}) {a=} {b=}")

        points = self.perimeter.points
        n = len(points)

        def make_unit_vector(i):
            pt = points[i % n]
            v = pt - center
            return v / np.linalg.norm(v)

        def closest_point_to_axis(axis, i):

            closest_idx = None
            closest_dot = 0
            
            for j in range(150):
                
                vec = make_unit_vector(i+j)
                dot = np.sum(vec * axis)
                if closest_dot < dot:
                    closest_idx = i+j
                    closest_dot = dot

                vec = make_unit_vector(i-j)
                dot = np.sum(vec * axis)
                if closest_dot < dot:
                    closest_idx = i-j
                    closest_dot = dot

            return closest_idx

        avg = make_unit_vector(b) + make_unit_vector(a)
        avg = avg / np.linalg.norm(avg)

        aa = closest_point_to_axis(avg, a)
        bb = closest_point_to_axis(avg, b)
        
        # print(f"  {aa=} {bb=}")

        tab['tangents'] = (aa, bb)

    @staticmethod
    def compute_convexity_defects(perimeter, approx_poly):
        points  = perimeter.points[approx_poly.indexes]
        convex_hull = cv.convexHull(points, returnPoints=False)
        return np.squeeze(cv.convexityDefects(points, convex_hull))

class EdgeComputer:

    def __init__(self, perimeter, approx_poly, tabs, verbose=True):

        self.perimeter = perimeter
        self.verbose = verbose
        self.edges = []

        len_hull = self.length_convex_hull(approx_poly)

        candidates = []
        for a, b in self.enumerate_candidates(approx_poly):
            l = self.length(a, b)
            candidates.append((l, a, b))

        candidates.sort(reverse=True)

        longest = candidates[0][0]

        if self.verbose:
            print(f"Identifying edges: length hull={len_hull:.1f}, longest candidate={longest:.1f} ...")

        for l, a, b in candidates:

            if self.verbose:
                print(f"considering candidate of length {l:.1f} from index {a} to {b}")
            
            if l < longest * 0.5:
                if self.verbose:
                    print(f"done: too short relative to longest candidate")
                break

            if l < len_hull * .10:
                if self.verbose:
                    print(f"done: too short relative to convex hull (ratio={l/len_hull:.3f})")
                break

            if self.overlaps_tab(tabs, a, b):
                if self.verbose:
                    print(f"skipping, overlaps one or more already identified tabs")
                continue

            points = self.points_for_line(a, b)
            line = np.squeeze(cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01))
            v, c = line[:2], line[2:]
            l *= .5
            pt0, pt1 = c - v*l, c + v*l

            # order points so that a -> b corresponds to pt0 -> pt1
            ptA = self.perimeter.points[a]
            if np.linalg.norm(pt1 - ptA) < np.linalg.norm(pt0 - ptA):
                pt0, pt1 = pt1, pt0
            line = puzzler.geometry.Line(np.array((pt0, pt1)))

            if self.is_tab_aligned(tabs, line):
                if self.verbose:
                    print(f"skipping, is approximately aligned with one or more tabs")
                continue

            if self.verbose:
                print(f"adding {line} from index {a} to {b}")
            
            self.edges.append({'fit_indexes':(a,b), 'line':line})

    def overlaps_tab(self, tabs, a, b):
        
        n = len(self.perimeter.points)

        def overlaps(i, j):

            a0, b0 = i
            if b0 < a0:
                return overlaps((a0, n), j) or overlaps((0, b0), j)
            
            a1, b1 = j
            if b1 < a1:
                return overlaps(i, (a1, n)) or overlaps(i, (0, b1))

            return a1 < b0 and a0 < b1
        
        for tab in tabs:

            if overlaps((a, b), tab['indexes']):
                return True

        return False

    def is_tab_aligned(self, tabs, line):

        v = line.pts[1] - line.pts[0]
        v = v / np.linalg.norm(v)
        line_normal = np.array((-v[1], v[0]))

        for t in tabs:
            tab_normal = self.get_tab_direction(t)
            dot_product = np.sum(line_normal * tab_normal)
            if dot_product > 0.7:
                return True

        return False

    def get_tab_direction(self, t):
        v = self.perimeter.points[np.array(t['tangents'])] - t['ellipse'].center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t['indent']:
            v = -v
        return v

    def points_for_line(self, a, b):

        points = self.perimeter.points
        if a < b:
            return points[a:b+1]
        
        return np.vstack((points[a:], points[:b+1]))
            
    def enumerate_candidates(self, approx_poly):
        
        runs = approx_poly.curvature_runs()

        if not runs[0][0] and not runs[-1][0]:
            a = runs[-1][1][-1]
            b = runs[0][1][0]
            yield (a,b)
        
        for in_out, indices in runs:

            if in_out:
                continue

            for a, b in zip(indices, indices[1:]):
                yield (a,b)
        
    def length(self, a, b):

        p0 = self.perimeter.points[a]
        p1 = self.perimeter.points[b]
        return np.linalg.norm(p0 - p1)

    def length_convex_hull(self, approx_poly):
        points  = self.perimeter.points[approx_poly.indexes]
        convex_hull = np.squeeze(cv.convexHull(points, returnPoints=True))
        diff = np.diff(convex_hull, axis=0, append=convex_hull[0:1])
        return np.sum(np.linalg.norm(diff, axis=1))

class EllipseFitterTk:

    def __init__(self, root, puzzle, label):
        
        piece = None
        for p in puzzle.pieces:
            if p.label == label:
                piece = p
        assert piece is not None

        self.label = label
        self.perimeter = PerimeterLoader(piece)
        self.convex_hull = None
        self.approx_pts = None
        self.convexity_defects = None
        self.tabs = None
        self.edges = None

        self.run(root)

    def approx_poly(self):

        apc = ApproxPolyComputer(self.perimeter, self.var_epsilon.get())

        points = self.perimeter.points

        convex_hull = cv.convexHull(points, returnPoints=False)
        self.convex_hull = np.squeeze(points[convex_hull])

        # print(f"{points=}")
        # print(f"{convex_hull=}")

        self.convexity_defects = []
        convexity_defects = TabComputer.compute_convexity_defects(self.perimeter, apc)

        # print(f"{convexity_defects=}")
        for defect in convexity_defects:
            indexes = apc.indexes
            p0 = points[indexes[defect[0]]]
            p1 = points[indexes[defect[1]]]
            p2 = points[indexes[defect[2]]]
            self.convexity_defects.append(np.array((p0, p2, p1)))

        self.approx_pts = np.array([points[i] for i in apc.indexes])
        self.signed_area = apc.signed_area

        self.render()

    def ellipsify(self):
        tc = TabComputer(self.perimeter, self.var_epsilon.get())
        self.tabs = tc.tabs

        ec = EdgeComputer(self.perimeter, tc.approx_poly, tc.tabs)
        self.edges = ec.edges

        self.render()

        print(f"tabs={self.tabs}\nedges={self.edges}")

        if len(self.edges) == 2:
            l0, l1 = self.edges[0]['line'], self.edges[1]['line']
            print(f"{l0=} {l1=}")
            v0 = l0.pts[1] - l0.pts[0]
            v0 = v0 / np.linalg.norm(v0)
            v1 = l1.pts[1] - l1.pts[0]
            v1 = v1 / np.linalg.norm(v1)
            print(f"{v0=} {v1=}")
            cross = np.cross(v0, v1)
            print(f"{cross=}")

    def get_camera_matrix(self):

        t = puzzler.render.Transform()
        
        h = int(self.canvas.configure('height')[4])
        camera_matrix = np.array(
            ((1,  0,   0),
             (0, -1, h-1),
             (0,  0,   1)), dtype=np.float64)
        t.multiply(camera_matrix)
        t.scale(self.camera_scale)
        t.translate(-self.camera_trans)

        return t.matrix

    def render(self):

        canvas = self.canvas
        canvas.delete('all')

        r = puzzler.renderer.canvas.CanvasRenderer(self.canvas)

        r.transform(self.get_camera_matrix())

        if self.var_render_perimeter.get():
            r.draw_points(self.perimeter.points, radius=1, fill='black')

        if self.var_render_convex_hull.get():
            if self.convex_hull is not None:
                r.draw_polygon(self.convex_hull, outline='black', fill='', width=1)

        if self.var_render_ellipse_points.get():
            if self.tabs is not None:
                for i, tab in enumerate(self.tabs):
                    points = self.perimeter.slice(*tab['indexes'])
                    r.draw_points(points, radius=4, fill='purple', outline='')

        if self.var_render_approx_poly.get():
            if self.approx_pts is not None:
                r.draw_polygon(self.approx_pts, outline='#00ff00', width=2, fill='')

        if self.var_render_curvature.get():
            if self.approx_pts is not None and self.signed_area is not None:
                for xy, area in zip(self.approx_pts, self.signed_area):
                    color = 'red' if area >= 0 else 'blue'
                    r.draw_points(xy, radius=4, fill=color, outline='')

        if self.var_render_approx_poly_index.get():
            if self.approx_pts is not None:
                for i, xy in enumerate(self.approx_pts):
                    r.draw_text(xy, text=f"{i}", fill='green')

        if self.var_render_convexity_defects.get():
            if self.convexity_defects is not None:
                for defect in self.convexity_defects:
                    r.draw_lines(defect, fill='lightblue', width=1)

        if self.var_render_ellipses.get():
            if self.tabs is not None:
                for i, tab in enumerate(self.tabs):
                    pts = puzzler.geometry.get_ellipse_points(tab['ellipse'], npts=40)
                    r.draw_lines(pts, fill='blue', width=2)
                    center = tab['ellipse'].center
                    r.draw_text(center, text=f"{i}", fill='red')
                    for j in tab['tangents']:
                        if j is None:
                            # print("FNORD!")
                            continue
                        p = self.perimeter.points[j]
                        r.draw_points(p, radius=6, fill='green')
                        r.draw_lines(np.array((center, p)), fill='green')
                    for j in tab['trimmed_indexes']:
                        p = self.perimeter.points[j]
                        r.draw_points(p, radius=6, fill='cyan')

        if self.var_render_lines.get():
            if self.edges is not None:
                for edge in self.edges:
                    line = edge['line']
                    r.draw_lines(line.pts, fill='blue', width='2')

    def motion(self, event):
        cm = self.get_camera_matrix()
        xy0 = np.array((event.x, event.y, 1))
        xy1 = xy0 @ np.linalg.inv(cm).T

        xy = xy1[:2]
        x2y2 = np.square(self.perimeter.points - xy)
        d = np.sum(x2y2, axis=1)
        ii = np.argmin(d)

        s = f"{xy[0]:.0f},{xy[1]:.0f}"
        if d[ii] < 25:
            s += f": point={ii}"

        self.var_label.set(s)
        
    def _init_controls(self, parent):
    
        self.controls = ttk.Frame(parent)
        self.controls.grid(column=0, row=1, sticky=(N, W, E, S), pady=5)

        cf1 = ttk.LabelFrame(self.controls, text='Render')
        cf1.grid(column=0, row=0, sticky=(N,W))
        
        self.var_render_perimeter = IntVar(value=1)
        cb1 = ttk.Checkbutton(cf1, text='Perimeter', command=self.render,
                              variable=self.var_render_perimeter)
        cb1.grid(column=0, row=0)

        self.var_render_convex_hull = IntVar(value=0)
        cb2 = ttk.Checkbutton(cf1, text='Convex Hull', command=self.render,
                              variable=self.var_render_convex_hull)
        cb2.grid(column=1, row=0)

        self.var_render_convexity_defects = IntVar(value=0)
        cb3 = ttk.Checkbutton(cf1, text='Defects', command=self.render,
                              variable=self.var_render_convexity_defects)
        cb3.grid(column=2, row=0)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=40)
        l1.grid(column=1, row=0)
        
        cf2 = ttk.LabelFrame(self.controls, text='Approx')
        cf2.grid(column=0, row=1, sticky=(N,W))
        b1 = ttk.Button(cf2, text='Approx', command=self.approx_poly)
        b1.grid(column=0, row=0)

        l1 = ttk.Label(cf2, text='Epsilon')
        l1.grid(column=1, row=0)

        self.var_epsilon = DoubleVar(value=10.)
        e1 = ttk.Entry(cf2, width=8, textvariable=self.var_epsilon)
        e1.grid(column=2, row=0)

        self.var_render_approx_poly = IntVar(value=0)
        cb4 = ttk.Checkbutton(cf2, text='Approx Poly', command=self.render,
                              variable=self.var_render_approx_poly)
        cb4.grid(column=3, row=0)

        self.var_render_curvature = IntVar(value=1)
        cb5 = ttk.Checkbutton(cf2, text='Curvature', command=self.render,
                              variable=self.var_render_curvature)
        cb5.grid(column=4, row=0)

        self.var_render_approx_poly_index = IntVar(value=0)
        cb6 = ttk.Checkbutton(cf2, text='Indexes', command=self.render,
                              variable=self.var_render_approx_poly_index)
        cb6.grid(column=5, row=0)

        cf3 = ttk.LabelFrame(self.controls, text='Tabs')
        cf3.grid(column=0, row=2, sticky=(N,W))
        b2 = ttk.Button(cf3, text='Ellipsify', command=self.ellipsify)
        b2.grid(column=0, row=0)

        self.var_render_ellipse_points = IntVar(value=1)
        cb7 = ttk.Checkbutton(cf3, text='Points', command=self.render,
                              variable=self.var_render_ellipse_points)
        cb7.grid(column=1, row=0)

        self.var_render_ellipses = IntVar(value=1)
        cb8 = ttk.Checkbutton(cf3, text='Ellipses', command=self.render,
                              variable=self.var_render_ellipses)
        cb8.grid(column=2, row=0)

        self.var_render_lines = IntVar(value=1)
        cb9 = ttk.Checkbutton(cf3, text='Lines', command=self.render,
                              variable=self.var_render_lines)
        cb9.grid(column=3, row=0)

    def run(self, parent):

        bbox = list(self.perimeter.bbox)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        bbox[0] -= w // 5
        bbox[2] += w // 5
        bbox[1] -= h // 5
        bbox[3] += h // 5
        
        s = 1
        if max(w,h) > 800:
            s = 800 / max(w,h)
            w = int(w*s)
            h = int(h*s)

        # print(f"{w=} {h=} {s=}")
        self.camera = bbox
        self.camera_trans = np.array(bbox[0:2])
        self.camera_scale = min(w / (bbox[2]-bbox[0]), h / (bbox[3]-bbox[1]))
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.canvas = Canvas(self.frame, width=w, height=h,
                             background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Motion>", self.motion)

        self._init_controls(self.frame)

        self.render()

def feature_view(args):

    puzzle = puzzler.file.load(args.puzzle)

    root = Tk()
    ui = EllipseFitterTk(root, puzzle, args.label)
    ui.var_epsilon.set(args.epsilon)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: features view")
    root.wm_resizable(0, 0)
    root.mainloop()

def feature_update(args):

    tab_data = []
    puzzle = puzzler.file.load(args.puzzle)
    for piece in tqdm(puzzle.pieces, ascii=True):
        if piece.points is None:
            continue

        # print(piece.label)

        perimeter = PerimeterLoader(piece)

        tc = TabComputer(perimeter, args.epsilon, False)
        ec = EdgeComputer(perimeter, tc.approx_poly, tc.tabs, False)

        piece.tabs = []
        for i, tab in enumerate(sorted(tc.tabs, key=operator.itemgetter('indexes'))):
            piece.tabs.append(puzzler.feature.Tab(tab['indexes'], tab['ellipse'], tab['indent'], tab['tangents']))
            tab_data.append((piece.label, i, tab['mse']))

        # print(" tabs:", ', '.join(f"{t.ellipse.semi_major/t.ellipse.semi_minor:.3f}" for t in piece.tabs))
                
        piece.edges = []
        for edge in sorted(ec.edges, key=operator.itemgetter('fit_indexes')):
            piece.edges.append(puzzler.feature.Edge(edge['fit_indexes'], edge['line']))
            
    puzzler.file.save(args.puzzle, puzzle)

    print("piece,tab_no,mse")
    for label, tab_no, mse in tab_data:
        print(f"{label},{tab_no},{mse:.3f}")

def add_parser(commands):
    
    parser_features = commands.add_parser("features", help="featurify pieces")
    parser_features.add_argument("-e", "--epsilon", default=10.0, type=float, help="epsilon for approximating polygon")

    commands = parser_features.add_subparsers()

    parser_update = commands.add_parser("update", help="update the feature computation")
    parser_update.set_defaults(func=feature_update)
    
    parser_view = commands.add_parser("view", help="view the feature computation")
    parser_view.add_argument("label")
    parser_view.set_defaults(func=feature_view)
