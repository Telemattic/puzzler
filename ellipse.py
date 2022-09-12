import argparse
import bisect
import cv2
import itertools
import math
import numpy as np
import PySimpleGUI as sg

# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

class PerimeterComputer:

    def __init__(self, filename):
        self.points = self._compute_points(filename)
        self.index  = self._init_index(self.points)

    @staticmethod
    def _compute_points(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 2
        return max(contours[0], key=cv2.contourArea)

    @staticmethod
    def _init_index(points):
        return dict((tuple(xy), i) for i, xy in enumerate(np.squeeze(points)))

class ApproxPolyComputer:

    def __init__(self, perimeter, epsilon):
        self.epsilon = epsilon
        self.indexes = self._compute_poly(perimeter, epsilon)
        self.signed_area = self._compute_signed_areas(perimeter, self.indexes)

    @staticmethod
    def _compute_poly(perimeter, epsilon):
        
        approx = cv2.approxPolyDP(perimeter.points, epsilon, True)
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

    def __init__(self, filename, epsilon):
        
        self.filename = filename
        self.perimeter = PerimeterComputer(filename)
        self.approx_poly = ApproxPolyComputer(self.perimeter, epsilon)

        self.curvature_runs()
        
        self.ellipses = []

        for defect in self.compute_convexity_defects():

            if defect[3] < 8000:
                continue
            
            l, r, c = defect[0], defect[1], defect[2]
            ellipse = self.fit_ellipse(l, r, c)
            if ellipse is not None:
                self.ellipses.append(ellipse)

    def curvature_runs(self):

        n = len(self.approx_poly.indexes)
        signs = [self.approx_poly.signed_area[i] > 0 for i in range(n)]
        print(f"{signs=}")

        signs = [(k,len(list(g))) for k, g in itertools.groupby(signs)]
        print(f"{signs=}")

    def fit_ellipse(self, l, r, c):

        print(f"fit_ellipse: {l=} {r=} {c=}")

        indexes, signed_area = self.approx_poly.indexes, self.approx_poly.signed_area

        i = bisect.bisect_left(indexes, c)

        a = i
        while l < indexes[a] and signed_area[a] > 0:
            a -= 1

        b = i
        while indexes[b] < r and signed_area[b] > 0:
            b += 1

        print(f"  approx: {i=} {a=} {b=}")
        
        a = indexes[a]
        b = indexes[b]
        
        print(f"  perimeter: {a=} {b=}")

        x = np.asarray(self.perimeter.points[a:b,0,0], dtype=np.float32)
        y = np.asarray(self.perimeter.points[a:b,0,1], dtype=np.float32)

        # print(f"  {x=} {y=}")

        coeffs = None
        try:
            coeffs = fit_ellipse(x, y)
        except np.linalg.LinAlgError as err:
            print("  LinAlgError: {0}".format(err))
            return None

        # throw away complex results, wtf is going on here anyway?
        if coeffs.dtype.kind == 'c':
            print("  complex result?")
            return None

        poly = cart_to_pol(coeffs)

        points = list(zip(x,y))

        cx, cy = poly[0], poly[1]
        x0, y0 = points[0]
        x1, y1 = points[-1]

        angle0 = math.atan2(y0-cy, x0-cx) * 180. / math.pi
        angle1 = math.atan2(y1-cy, x1-cx) * 180. / math.pi
        angles = [angle0, angle1]

        diff = angle1 - angle0
        if diff < 0:
            diff += 360.

        print(f"  angles={angle0:.1f},{angle1:.1f} -> {diff:.1f}")
        
        return {'coeffs': coeffs, 'poly': poly, 'points': points, 'angles': angles}

    def compute_convexity_defects(self):
        convex_hull = cv2.convexHull(self.perimeter.points, returnPoints=False)
        return np.squeeze(cv2.convexityDefects(self.perimeter.points, convex_hull))

class EllipseFitter:

    def __init__(self, filename):
        self.filename = filename
        self.epsilon  = 1.
        self.perimeter = PerimeterComputer(filename)
        self.convex_hull = None
        self.approx_pts = None
        self.convexity_defects = None
        self.ellipses = None

    def approx_poly(self):

        apc = ApproxPolyComputer(self.perimeter, self.epsilon)

        convex_hull = cv2.convexHull(self.perimeter.points, returnPoints=False)
        self.convex_hull = np.squeeze(self.perimeter.points[convex_hull])

        self.convexity_defects = []
        convexity_defects = cv2.convexityDefects(self.perimeter.points, convex_hull)
        for defect in np.squeeze(convexity_defects):
            p0 = tuple(*self.perimeter.points[defect[0]])
            p1 = tuple(*self.perimeter.points[defect[1]])
            p2 = tuple(*self.perimeter.points[defect[2]])
            self.convexity_defects.append([p0, p2, p1])

        self.approx_pts = [tuple(*self.perimeter.points[i]) for i in apc.indexes]
        self.signed_area = apc.signed_area

        self.render()

    def ellipsify(self):
        tab_computer = TabComputer(self.filename, self.epsilon)
        self.ellipses = tab_computer.ellipses
        
        self.render()

    def render(self):

        graph = self.window['graph']
        graph.erase()

        if self.window['render_image'].get():
            graph.draw_image(filename=self.filename, location=(0,0))

        if self.window['render_perimeter'].get():
                for xy in self.perimeter.points:
                    graph.draw_point(tuple(*xy), size=1, color='yellow')

        if self.window['render_convex_hull'].get():
            if self.convex_hull is not None:
                xy_tuples = list(tuple(i) for i in self.convex_hull)
                xy_tuples.append(xy_tuples[0])
                graph.draw_lines(xy_tuples, color='yellow', width=1)

        if self.window['render_ellipse_points'].get():
            if self.ellipses is not None:
                for i, ellipse in enumerate(self.ellipses):
                    for p in ellipse['points']:
                        graph.draw_point(p, size=8, color='purple')

        if self.window['render_approx_poly'].get():
            if self.approx_pts is not None:
                graph.draw_lines(self.approx_pts, color='#00ff00', width=2)

        if self.window['render_curvature'].get():
            if self.approx_pts is not None and self.signed_area is not None:
                for xy, area in zip(self.approx_pts, self.signed_area):
                    color = 'red' if area >= 0 else 'blue'
                    graph.draw_point(xy, size=5, color=color)

        if self.window['render_approx_poly_index'].get():
            if self.approx_pts is not None:
                for i, xy in enumerate(self.approx_pts):
                    graph.draw_text(f"{i}", xy, color='green')

        if self.window['render_convexity_defects'].get():
            if self.convexity_defects is not None:
                for defect in self.convexity_defects:
                    graph.draw_lines(defect, color='lightblue', width=1)

        if self.window['render_ellipses'].get():
            if self.ellipses is not None:
                for i, ellipse in enumerate(self.ellipses):
                    poly = ellipse['poly']
                    angles = ellipse['angles']
                    print(f"{i}: x,y={poly[0]:.1f},{poly[1]:.1f} angles={angles[0]:.1f},{angles[1]:.1f}")
                    pts = get_ellipse_pts(poly, npts=20)
                    pts = list(zip(pts[0], pts[1]))
                    # print(f"  {pts=}")
                    graph.draw_lines(pts, color='blue', width=2)
                    graph.draw_text(f"{i}", (poly[0], poly[1]), color='red')

    def ui(self):

        render_layout = [
            sg.CB('Image',       default=True, enable_events=True, key='render_image'),
            sg.CB('Perimeter',   default=True, enable_events=True, key='render_perimeter'),
            sg.CB('Convex Hull', default=True, enable_events=True, key='render_convex_hull'),
            sg.CB('Defects',     default=True, enable_events=True, key='render_convexity_defects')
        ]

        approx_layout = [
            sg.Button('Approx', key='button_approx'),
            sg.Text('Epsilon'),
            sg.InputText('1.0', key='epsilon', size=(5,1), enable_events=True),
            sg.CB('Approx Poly', default=True, enable_events=True, key='render_approx_poly'),
            sg.CB('Curvature',   default=True, enable_events=True, key='render_curvature'),
            sg.CB('Indexes', default=True, enable_events=True, key='render_approx_poly_index')
        ]

        tabs_layout = [
            sg.Button('Ellipsify', key='button_ellipsify'),
            sg.CB('Points', default=True, enable_events=True, key='render_ellipse_points'),
            sg.CB('Ellipses', default=True, enable_events=True, key='render_ellipses')
        ]
        
        layout = [
            [sg.Graph(canvas_size=(551,551),
                      graph_bottom_left = (0,550),
                      graph_top_right = (550,0),
                      background_color='black',
                      key='graph',
                      drag_submits=True,
                      enable_events=True,
                      metadata=self)],
            [sg.Frame('Render', [render_layout])],
            [sg.Frame('Approx', [approx_layout])],
            [sg.Frame('Tabs',   [tabs_layout])]
        ]
        self.window = sg.Window('Ellipse Fitter', layout, finalize=True)
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
                self.epsilon = float(self.window['epsilon'].get())
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("image")

    args = parser.parse_args()

    e = EllipseFitter(args.image)
    e.ui()

if __name__ == '__main__':
    main()
