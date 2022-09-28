import bisect
import cv2 as cv
import itertools
import math
import numpy as np
import operator
import PySimpleGUI as sg
import puzzler

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

        self.tabs = []

        for defect in self.compute_convexity_defects(self.perimeter, self.approx_poly):

            if defect[3] < 8000:
                continue
            
            l, r, c = defect[0], defect[1], defect[2]
            tab = self.fit_ellipse_to_convexity_defect(l, r, c)
            if tab is None:
                continue

            if tab['angle'] < math.radians(220):
                if self.verbose:
                    print("  angle for ellipse too small, rejecting")
                continue
            
            self.tabs.append(tab)

        for in_out, indices in self.approx_poly.curvature_runs():

            if in_out:
                continue
            
            tab = self.fit_ellipse_to_outdent(indices[0], indices[-1])
            if tab is None:
                continue

            if tab['angle'] < math.radians(220):
                if self.verbose:
                    print("  angle for ellipse too small, rejecting")
                continue
            
            if self.indexes_overlap(tab):
                if self.verbose:
                    print("outdent tab overlaps previously found tab")
                continue
            
            self.tabs.append(tab)

        for _ in range(3):
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
            print(f"fit_ellipse_to_outdent: {l=} {r=}")

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
            print(f"fit_ellipse_to_convexity_defect: {l=} {r=} {c=}")

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

        ellipse = puzzler.geometry.fit_ellipse_to_points(self.perimeter.slice(a, b))
        if ellipse is None:
            return None

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

        tab = {'ellipse': ellipse, 'indexes': (a,b), 'indent':indent, 'angle': angle}

        # indices of last two points that are "close enough" to the ellipse
        ## aa, bb = self.find_fit_range(ellipse)

        # indices of tangent points
        self.find_tangent_points(tab)

        self.trim_indexes(tab)

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

    @staticmethod
    def distance_to_ellipse(semi_major, semi_minor, p):
        px = abs(p[0])
        py = abs(p[1])

        tx = 0.707
        ty = 0.707

        a = semi_major
        b = semi_minor

        for x in range(0, 3):
            x = a * tx
            y = b * ty

            ex = (a*a - b*b) * tx**3 / a
            ey = (b*b - a*a) * ty**3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(rx, ry)
            q = math.hypot(qx, qy)

            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
            t = math.hypot(tx, ty)
            tx /= t 
            ty /= t 

        # return (math.copysign(a * tx, p[0]), math.copysign(b * ty, p[1])
        return math.hypot(a * tx - px, b * ty - py)

    def trim_indexes(self, tab):

        ellipse = tab['ellipse']
        center = ellipse.center
        semi_major, semi_minor = ellipse.semi_major, ellipse.semi_minor
        angle = ellipse.phi
        c, s = math.cos(angle), math.sin(angle)

        global_to_local = np.array((( c, s),
                                    (-s, c)))

        # print(f"trim_indexes: center=({center[0]=:.1f},{center[1]=:.1f}) {semi_major=:.1f} {semi_minor=:.1f} {angle=:.3f}")
        # print(f"  {global_to_local=}")

        a, b = tab['indexes']
        n = len(self.perimeter.points)
        
        for aa in range(a,a+50):
            ptg = self.perimeter.points[aa % n]
            ptl = (ptg - center) @ global_to_local
            d = self.distance_to_ellipse(semi_major, semi_minor, ptl)
            # print(f"   {aa=} {ptg=} {ptl=} {d=:.1f}")
            if d < 5:
                break

        for bb in range(b,b-50,-1):
            ptg = self.perimeter.points[bb]
            ptl = (ptg - center) @ global_to_local
            d = self.distance_to_ellipse(semi_major, semi_minor, ptl)
            # print(f"   {bb=} {ptg=} {ptl=} {d=:.1f}")
            if d < 5:
                break

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

    def __init__(self, perimeter, approx_poly):

        self.perimeter = perimeter
        self.edges = []

        candidates = []
        for a, b in self.enumerate_candidates(approx_poly):
            l = self.length(a, b)
            candidates.append((l, a, b))

        candidates.sort(reverse=True)

        longest = candidates[0][0]
        for l, a, b in candidates:
            if l < longest * 0.5:
                break

            points = self.points_for_line(a, b)
            line = np.squeeze(cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01))
            v, c = line[:2], line[2:]
            l *= .5
            self.edges.append({'fit_indexes':(a,b), 'line':puzzler.geometry.Line(c-v*l, c+v*l)})

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

class EllipseFitter:

    def __init__(self, puzzle, label):
        
        piece = None
        for p in puzzle.pieces:
            if p.label == label:
                piece = p
        assert piece is not None

        self.label = label
        self.perimeter = PerimeterLoader(piece)
        self.epsilon  = 10.
        self.convex_hull = None
        self.approx_pts = None
        self.convexity_defects = None
        self.tabs = None
        self.edges = None

    def approx_poly(self):

        apc = ApproxPolyComputer(self.perimeter, self.epsilon)

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
            p0 = tuple(points[indexes[defect[0]]])
            p1 = tuple(points[indexes[defect[1]]])
            p2 = tuple(points[indexes[defect[2]]])
            self.convexity_defects.append([p0, p2, p1])

        self.approx_pts = [tuple(points[i]) for i in apc.indexes]
        self.signed_area = apc.signed_area

        self.render()

    def ellipsify(self):
        tab_computer = TabComputer(self.perimeter, self.epsilon)
        self.tabs = tab_computer.tabs

        edge_computer = EdgeComputer(self.perimeter, tab_computer.approx_poly)
        self.edges = edge_computer.edges

        self.render()

        print(f"tabs={self.tabs}\nedges={self.edges}")

    def render(self):

        graph = self.window['graph']
        graph.erase()

        if self.window['render_perimeter'].get():
                for xy in self.perimeter.points:
                    graph.draw_point(tuple(xy), size=1, color='black')

        if self.window['render_convex_hull'].get():
            if self.convex_hull is not None:
                xy_tuples = list(tuple(i) for i in self.convex_hull)
                xy_tuples.append(xy_tuples[0])
                graph.draw_lines(xy_tuples, color='black', width=1)

        if self.window['render_ellipse_points'].get():
            if self.tabs is not None:
                for i, tab in enumerate(self.tabs):
                    for p in self.perimeter.slice(*tab['indexes']):
                        graph.draw_point(p.tolist(), size=8, color='purple')

        if self.window['render_approx_poly'].get():
            if self.approx_pts is not None:
                graph.draw_lines(self.approx_pts + [self.approx_pts[0]], color='#00ff00', width=2)

        if self.window['render_curvature'].get():
            if self.approx_pts is not None and self.signed_area is not None:
                for xy, area in zip(self.approx_pts, self.signed_area):
                    color = 'red' if area >= 0 else 'blue'
                    graph.draw_point(xy, size=12, color=color)

        if self.window['render_approx_poly_index'].get():
            if self.approx_pts is not None:
                for i, xy in enumerate(self.approx_pts):
                    graph.draw_text(f"{i}", xy, color='green', font=(16))

        if self.window['render_convexity_defects'].get():
            if self.convexity_defects is not None:
                for defect in self.convexity_defects:
                    graph.draw_lines(defect, color='lightblue', width=1)

        if self.window['render_ellipses'].get():
            if self.tabs is not None:
                for i, tab in enumerate(self.tabs):
                    pts = puzzler.geometry.get_ellipse_points(tab['ellipse'], npts=40)
                    graph.draw_lines(pts, color='blue', width=2)
                    center = tab['ellipse'].center
                    graph.draw_text(f"{i}", center.tolist(), color='red', font=('Courier', 12))
                    for j in tab['tangents']:
                        if j is None:
                            # print("FNORD!")
                            continue
                        p = self.perimeter.points[j].tolist()
                        graph.draw_point(p, size=10, color='green')
                        graph.draw_line(center.tolist(), p, color='green')
                    for j in tab['trimmed_indexes']:
                        p = self.perimeter.points[j].tolist()
                        graph.draw_point(p, size=10, color='cyan')

        if self.window['render_lines'].get():
            if self.edges is not None:
                for edge in self.edges:
                    line = edge['line']
                    graph.draw_line(line.pt0.tolist(), line.pt1.tolist(), color='blue', width='2')

    def run(self):

        render_layout = [
            sg.CB('Perimeter',   default=True, enable_events=True, key='render_perimeter'),
            sg.CB('Convex Hull', default=False, enable_events=True, key='render_convex_hull'),
            sg.CB('Defects',     default=False, enable_events=True, key='render_convexity_defects')
        ]

        approx_layout = [
            sg.Button('Approx', key='button_approx'),
            sg.Text('Epsilon'),
            sg.InputText(f"{self.epsilon}", key='epsilon', size=(5,1), enable_events=True),
            sg.CB('Approx Poly', default=False, enable_events=True, key='render_approx_poly'),
            sg.CB('Curvature',   default=True, enable_events=True, key='render_curvature'),
            sg.CB('Indexes', default=False, enable_events=True, key='render_approx_poly_index')
        ]

        tabs_layout = [
            sg.Button('Ellipsify', key='button_ellipsify'),
            sg.CB('Points', default=True, enable_events=True, key='render_ellipse_points'),
            sg.CB('Ellipses', default=True, enable_events=True, key='render_ellipses'),
            sg.CB('Lines', default=True, enable_events=True, key='render_lines'),
        ]

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

        # print(f"{w=} {h=} {s=}")
        
        layout = [
            [sg.Graph(canvas_size=(int(w * s), int(h * s)),
                      graph_bottom_left = (bbox[0],bbox[1]),
                      graph_top_right = (bbox[2],bbox[3]),
                      background_color='white',
                      key='graph',
                      enable_events=True)],
            [sg.Frame('Render', [render_layout])],
            [sg.Frame('Approx', [approx_layout])],
            [sg.Frame('Tabs',   [tabs_layout])]
        ]
        self.window = sg.Window(f"Ellipse Fitter ({self.label})", layout, finalize=True)
        self.render()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == 'button_approx':
                self.approx_poly()
            elif event == 'button_ellipsify':
                self.ellipsify()
            elif event.startswith('render_'):
                self.render()
            elif event == 'epsilon':
                try:
                    self.epsilon = float(self.window['epsilon'].get())
                except ValueError as err:
                    print(err)
            else:
                print(event, values)

def feature_view(args):

    puzzle = puzzler.file.load(args.puzzle)
    ui = EllipseFitter(puzzle, args.label)
    ui.epsilon = args.epsilon
    ui.run()

def feature_update(args):

    puzzle = puzzler.file.load(args.puzzle)
    for piece in puzzle.pieces:
        if piece.points is None:
            continue

        print(piece.label)

        perimeter = PerimeterLoader(piece)

        tc = TabComputer(perimeter, args.epsilon, False)
        ec = EdgeComputer(perimeter, tc.approx_poly)

        piece.tabs = []
        for tab in tc.tabs:
            piece.tabs.append(puzzler.feature.Tab(tab['indexes'], tab['ellipse'], tab['indent'], tab['tangents']))
                
        piece.edges = []
        for edge in ec.edges:
            piece.edges.append(puzzler.feature.Edge(edge['fit_indexes'], edge['line']))

    puzzler.file.save(args.puzzle, puzzle)

def add_parser(commands):
    
    parser_features = commands.add_parser("features", help="featurify pieces")
    parser_features.add_argument("-e", "--epsilon", default=10.0, type=float, help="epsilon for approximating polygon")

    commands = parser_features.add_subparsers()

    parser_update = commands.add_parser("update", help="update the feature computation")
    parser_update.set_defaults(func=feature_update)
    
    parser_view = commands.add_parser("view", help="view the feature computation")
    parser_view.add_argument("label")
    parser_view.set_defaults(func=feature_view)
