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
        points = np.array(puzzler.chain.ChainCode().decode(piece['points']))
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

    def __init__(self, perimeter, epsilon):
        
        self.perimeter = perimeter
        self.approx_poly = ApproxPolyComputer(self.perimeter, epsilon)

        self.ellipses = []

        for defect in self.compute_convexity_defects():

            if defect[3] < 8000:
                continue
            
            l, r, c = defect[0], defect[1], defect[2]
            ellipse = self.fit_ellipse_to_convexity_defect(l, r, c)
            if ellipse is None:
                continue
            
            self.ellipses.append(ellipse)

        for in_out, indices in self.approx_poly.curvature_runs():

            if in_out:
                continue
            
            ellipse = self.fit_ellipse_to_outdent(indices[0], indices[-1])
            if ellipse is None:
                continue

            if self.indexes_overlap(ellipse):
                print("outdent ellipse overlaps previously found ellipse")
                continue
            
            self.ellipses.append(ellipse)

        for _ in range(3):
            for i, ellipse in enumerate(self.ellipses):
                a, b = ellipse['trimmed_indexes']
                ellipse2 = self.fit_ellipse(a, b, ellipse['indent'])
                self.ellipses[i] = ellipse2

    def indexes_overlap(self, ellipse):
        
        n = len(self.perimeter.points)
        a0, b0 = ellipse['indexes']
        if b0 < a0:
            b0 += n
                
        for e in self.ellipses:
                
            a1, b1 = e['indexes']
            if b1 < a1:
                b1 += n
                    
            if a1 < b0 and a0 < b1:
                return True

        return False
            
    def fit_ellipse_to_outdent(self, l, r):
        
        print(f"fit_ellipse_to_outdent: {l=} {r=}")

        indexes, signed_area = self.approx_poly.indexes, self.approx_poly.signed_area

        a = bisect.bisect_left(indexes, l)
        while signed_area[a] < 0:
            a -= 1

        b = bisect.bisect_left(indexes, r)
        n = len(signed_area)
        while signed_area[b] < 0:
            b += 1
            if b == n:
                return None

        print(f"  approx: {a=} {b=}")

        return self.fit_ellipse(indexes[a], indexes[b], False)

    def fit_ellipse_to_convexity_defect(self, l, r, c):
    
        print(f"fit_ellipse_to_convexity_defect: {l=} {r=} {c=}")

        indexes, signed_area = self.approx_poly.indexes, self.approx_poly.signed_area

        i = bisect.bisect_left(indexes, c)

        a = i
        while l < indexes[a] and signed_area[a] > 0:
            a -= 1

        b = i
        while indexes[b] < r and signed_area[b] > 0:
            b += 1

        print(f"  approx: {i=} {a=} {b=}")

        return self.fit_ellipse(indexes[a], indexes[b], True)

    def fit_ellipse(self, a, b, indent):

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

        print(f"  center={cx:.1f},{cy:.1f} major={ellipse.semi_major:.1f} minor={ellipse.semi_minor:.1f} phi={math.degrees(ellipse.phi):.1f}")
        print(f"  angle={math.degrees(angle):.1f}")

        if angle < math.radians(220):
            print("  angle for ellipse too small, rejecting")
            return None

        tab = {'ellipse': ellipse, 'indexes': (a,b), 'indent':indent, 'angle': ab}

        # indices of last two points that are "close enough" to the ellipse
        ## aa, bb = self.find_fit_range(ellipse)

        # indices of tangent points
        self.find_tangent_points(tab)

        self.trim_indexes(tab)

        return tab

    def angle_between(self, center, a, b):

        pa = self.perimeter.points[a] - center
        pb = self.perimeter.points[b] - center
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

    def trim_indexes(self, ellipse):

        poly = ellipse['poly']
        center = np.array((poly[:2]))
        semi_major, semi_minor = poly[2], poly[3]
        angle = poly[5]
        c, s = math.cos(angle), math.sin(angle)

        global_to_local = np.array((( c, s),
                                    (-s, c)))

        # print(f"trim_indexes: center=({center[0]=:.1f},{center[1]=:.1f}) {semi_major=:.1f} {semi_minor=:.1f} {angle=:.3f}")
        # print(f"  {global_to_local=}")

        a, b = ellipse['indexes']
        
        for aa in range(a,a+50):
            ptg = self.perimeter.points[aa]
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

        ellipse['trimmed_indexes'] = (aa, bb)

    def find_tangent_points(self, ellipse):

        center = np.array(ellipse['poly'][0:2])

        a, b = ellipse['indexes']
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
            
            for j in range(50):
                
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

        aa = closest_point_to_axis(make_unit_vector(b), a)
        bb = closest_point_to_axis(make_unit_vector(a), b)
        
        # print(f"  {aa=} {bb=}")

        ellipse['tangents'] = (aa, bb)

    def compute_convexity_defects(self):
        convex_hull = cv.convexHull(self.perimeter.points, returnPoints=False)
        return np.squeeze(cv.convexityDefects(self.perimeter.points, convex_hull))

class LineComputer:

    def __init__(self, perimeter, approx_poly):

        self.perimeter = perimeter
        self.lines = []

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
            line = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = np.squeeze(line)
            l *= .5
            self.lines.append((x0-l*vx, y0-l*vy, x0+l*vx, y0+l*vy))

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
        for p in puzzle['pieces']:
            if p['label'] == label:
                piece = p
        assert piece is not None

        self.label = label
        self.perimeter = PerimeterLoader(piece)
        self.epsilon  = 10.
        self.convex_hull = None
        self.approx_pts = None
        self.convexity_defects = None
        self.ellipses = None
        self.lines = None

    def approx_poly(self):

        apc = ApproxPolyComputer(self.perimeter, self.epsilon)

        points = self.perimeter.points

        convex_hull = cv.convexHull(points, returnPoints=False)
        self.convex_hull = np.squeeze(points[convex_hull])

        # print(f"{points=}")
        # print(f"{convex_hull=}")

        self.convexity_defects = []
        convexity_defects = cv.convexityDefects(points, convex_hull)

        # print(f"{convexity_defects=}")
        for defect in np.squeeze(convexity_defects):
            p0 = tuple(points[defect[0]])
            p1 = tuple(points[defect[1]])
            p2 = tuple(points[defect[2]])
            self.convexity_defects.append([p0, p2, p1])

        self.approx_pts = [tuple(points[i]) for i in apc.indexes]
        self.signed_area = apc.signed_area

        self.render()

    def ellipsify(self):
        tab_computer = TabComputer(self.perimeter, self.epsilon)
        self.ellipses = tab_computer.ellipses

        line_computer = LineComputer(self.perimeter, tab_computer.approx_poly)
        self.lines = line_computer.lines

        self.render()

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
            if self.ellipses is not None:
                for i, ellipse in enumerate(self.ellipses):
                    for p in self.perimeter.slice(*ellipse['indexes']):
                        graph.draw_point(p.tolist(), size=8, color='purple')

        if self.window['render_approx_poly'].get():
            if self.approx_pts is not None:
                graph.draw_lines(self.approx_pts + [self.approx_pts[0]], color='#00ff00', width=2)

        if self.window['render_curvature'].get():
            if self.approx_pts is not None and self.signed_area is not None:
                for xy, area in zip(self.approx_pts, self.signed_area):
                    color = 'red' if area >= 0 else 'blue'
                    graph.draw_point(xy, size=7, color=color)

        if self.window['render_approx_poly_index'].get():
            if self.approx_pts is not None:
                for i, xy in enumerate(self.approx_pts):
                    graph.draw_text(f"{i}", xy, color='green', font=(16))

        if self.window['render_convexity_defects'].get():
            if self.convexity_defects is not None:
                for defect in self.convexity_defects:
                    graph.draw_lines(defect, color='lightblue', width=1)

        if self.window['render_ellipses'].get():
            if self.ellipses is not None:
                for i, ellipse in enumerate(self.ellipses):
                    poly = ellipse['poly']
                    angle = ellipse['angle']
                    print(f"{i}: x,y={poly[0]:7.1f},{poly[1]:7.1f} angle{math.degrees(angle):6.1f} indexes={ellipse['indexes']}")
                    pts = puzzler.geometry.get_ellipse_points(ellipse['ellipse'], npts=40)
                    # pts = list(zip(pts[0], pts[1]))
                    # print(f"  {pts=}")
                    graph.draw_lines(pts, color='blue', width=2)
                    graph.draw_text(f"{i}", (poly[0], poly[1]), color='red', font=('Courier', 12))
                    for j in ellipse['tangents']:
                        p = self.perimeter.points[j].tolist()
                        graph.draw_point(p, size=10, color='green')
                        c = poly[0:2]
                        graph.draw_line(c, p, color='green')
                    for j in ellipse['trimmed_indexes']:
                        p = self.perimeter.points[j].tolist()
                        graph.draw_point(p, size=10, color='cyan')

        if self.window['render_lines'].get():
            if self.lines is not None:
                for line in self.lines:
                    pt1 = (line[0], line[1])
                    pt2 = (line[2], line[3])
                    graph.draw_line(pt1, pt2, color='blue', width='2')

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

def ellipse_ui(args):

    puzzle = puzzler.file.load(args.puzzle)
    ui = EllipseFitter(puzzle, args.label)
    ui.epsilon = args.epsilon
    ui.run()

def add_parser(commands):
    
    parser_ellipse = commands.add_parser("ellipse", help="ellipsify pieces")
    parser_ellipse.add_argument("label")
    parser_ellipse.add_argument("-e", "--epsilon", default=10.0, type=float, help="epsilon for approximating polygon")
    parser_ellipse.set_defaults(func=ellipse_ui)
