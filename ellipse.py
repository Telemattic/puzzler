import argparse
import bisect
import cv2
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

class TabComputer:

    def __init__(self, filename, epsilon):
        
        self.filename = filename
        self.epsilon  = epsilon

        self.init_perimeter(filename)
        self.init_approx_poly(epsilon)
        
        self.ellipses = []

        print(f"{np.squeeze(self.perimeter)=}")
        print(f"{self.approx_poly=}")
        print(f"{self.compute_convexity_defects()=}")

        for defect in self.compute_convexity_defects():

            if defect[3] < 8000:
                continue
            
            l, r, c = defect[0], defect[1], defect[2]
            ellipse = self.fit_ellipse(l, r, c)
            if ellipse is not None:
                self.ellipses.append(ellipse)

        print(f"{self.ellipses=}")

    def fit_ellipse(self, l, r, c):

        print(f"fit_ellipse: {l=} {r=} {c=}")

        i = bisect.bisect(self.approx_poly, c)

        a = i
        while l < self.approx_poly[a] and self.signed_area(a) > 0:
            a -= 1

        b = i
        while self.approx_poly[b] < r and self.signed_area(b) > 0:
            b += 1

        print(f"  approx: {i=} {a=} {b=}")
        
        a = self.approx_poly[a]
        b = self.approx_poly[b]
        
        print(f"  perimeter: {a=} {b=}")

        x = np.asarray(self.perimeter[a:b,0,0], dtype=np.float32)
        y = np.asarray(self.perimeter[a:b,0,1], dtype=np.float32)
        print(f"  {x=} {y=}")

        coeffs = fit_ellipse(x, y)

        # throw away complex results, wtf is going on here anyway?
        if coeffs.dtype.kind == 'c':
            return None

        poly = cart_to_pol(coeffs)

        points = list(zip(x,y))

        cx, cy = poly[0], poly[1]
        x0, y0 = points[0]
        x1, y1 = points[-1]

        angle0 = math.atan2(y0-cy, x0-cx) * 180. / math.pi
        angle1 = math.atan2(y1-cy, x1-cx) * 180. / math.pi
        angles = [angle0, angle1]

        print(f"{angles=}")
        
        return {'coeffs': coeffs, 'poly': poly, 'points': points, 'angles': angles}

    def signed_area(self, i):
        x0, y0 = self.perimeter[self.approx_poly[i-1]][0]
        x1, y1 = self.perimeter[self.approx_poly[i]][0]
        x2, y2 = self.perimeter[self.approx_poly[i+1]][0]
        return (x1-x0) * (y2-y1) - (x2-x1)*(y1-y0)

    def init_perimeter(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 2
        self.perimeter = max(contours[0], key=cv2.contourArea)
        self.point_to_index = dict()
        for i, xy in enumerate(np.squeeze(self.perimeter)):
            self.point_to_index[tuple(xy)] = i

    def compute_convexity_defects(self):
        convex_hull = cv2.convexHull(self.perimeter, returnPoints=False)
        return np.squeeze(cv2.convexityDefects(self.perimeter, convex_hull))

    def init_approx_poly(self, epsilon):
        approx = cv2.approxPolyDP(self.perimeter, epsilon, True)
        poly = list(self.point_to_index[tuple(xy)] for xy in np.squeeze(approx))

        reset = None
        for i in range(1,len(poly)):
            if poly[i-1] > poly[i]:
                assert reset is None
                reset = i
            else:
                assert poly[i-1] < poly[i]
                
        if reset:
            poly = poly[reset:] + poly[:reset]

        self.approx_poly = poly

class EllipseFitter:

    def __init__(self, filename):
        self.filename = filename
        self.ellipse_pts = None
        self.mouse_rect = None
        self.approx_pts = None
        self.perimeter_pts = None
        self.convex_hull = None
        self.convexity_defects = None
        self.ellipses = None

    def fit_ellipse_to_rect(self, rect):

        print(f"fit_ellipse_to_rect: {rect=}")

        x0, y0 = rect[0]
        x1, y1 = rect[1]
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1

        if self.perimeter_pts is not None:

            x_vec = []
            y_vec = []
            for x, y in self.perimeter_pts:

                if x0 <= x and x <= x1 and y0 <= y and y <= y1:
                    x_vec.append(x-x0)
                    y_vec.append(y-y0)

            x_vec = np.asarray(x_vec, dtype=np.float32)
            y_vec = np.asarray(y_vec, dtype=np.float32)

        else:

            img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            image_pts = np.nonzero(img[y0:y1, x0:x1])
            x_vec = image_pts[1]
            y_vec = image_pts[0]

        print(f"{x_vec=} {y_vec=}")

        coeffs = fit_ellipse(x_vec, y_vec)

        print(f"{coeffs=}")

        ellipse_pts = get_ellipse_pts(cart_to_pol(coeffs))

        ret = list(zip(ellipse_pts[0]+x0, ellipse_pts[1]+y0))

        return ret

    def approx_poly(self, event, values):

        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 2
        c = max(contours[0], key=cv2.contourArea)

        convex_hull = cv2.convexHull(c, returnPoints=False)
        self.convex_hull = np.squeeze(c[convex_hull])
        # print(f"{self.convex_hull=}")

        self.convexity_defects = []
        convexity_defects = cv2.convexityDefects(c, convex_hull)
        for defect in np.squeeze(convexity_defects):
            p0 = tuple(*c[defect[0]])
            p1 = tuple(*c[defect[1]])
            p2 = tuple(*c[defect[2]])
            self.convexity_defects.append([p0, p2, p1])
        # print(f"{self.convexity_defects=}")

        self.perimeter_pts = list(tuple(i[0]) for i in c)

        eps = float(self.window['epsilon'].get())

        approx_pts = cv2.approxPolyDP(c, eps, True)
        # print(f"{len(approx_pts)=} {approx_pts=}")

        self.approx_pts = approx_pts

        self.render()

    def ellipsify(self, event, values):
        epsilon = float(self.window['epsilon'].get())
        tab_computer = TabComputer(self.filename, epsilon)
        self.ellipses = tab_computer.ellipses
        self.render()

    def handle_mouse(self, event, values):

        xy = values['graph']
        
        if self.mouse_rect is None:
            self.mouse_rect = [xy, xy]
        else:
            self.mouse_rect[1] = xy
        
        if event == 'graph+UP':
            self.ellipse_pts = self.fit_ellipse_to_rect(self.mouse_rect)
            self.mouse_rect = None

        self.render()

    def render(self):

        graph = self.window['graph']
        graph.erase()

        id = graph.draw_image(filename=self.filename, location=(0,0))
        print(f"draw_image: {id=}")

        if self.perimeter_pts is not None:
            for xy in self.perimeter_pts:
                graph.draw_point(xy, size=1, color='yellow')

        if self.convex_hull is not None:
            xy_tuples = list(tuple(i) for i in self.convex_hull)
            xy_tuples.append(xy_tuples[0])
            graph.draw_lines(xy_tuples, color='yellow', width=1)

        if self.ellipses is not None:
            for i, ellipse in enumerate(self.ellipses):
                for p in ellipse['points']:
                    graph.draw_point(p, size=8, color='purple')
                    
        if self.approx_pts is not None:
            xy_tuples = list(tuple(i[0]) for i in self.approx_pts)
            graph.draw_lines(xy_tuples, color='#00ff00', width=2)

            for i in range(len(xy_tuples)):
                x0, y0 = xy_tuples[i-2]
                x1, y1 = xy_tuples[i-1]
                x2, y2 = xy_tuples[i]
                area = (x1-x0) * (y2-y1) - (x2-x1)*(y1-y0)
                color = 'red' if area >= 0 else 'blue'
                graph.draw_point((x1,y1), size=5, color=color)

        if self.convexity_defects is not None:
            for defect in self.convexity_defects:
                graph.draw_lines(defect, color='lightblue', width=1)

        if self.ellipse_pts:
            graph.draw_lines(self.ellipse_pts, color='blue', width=2)

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

        if self.mouse_rect:
            tl, br = self.mouse_rect[0], self.mouse_rect[1]
            graph.draw_rectangle(tl, br, line_color='red', line_width=2)

    def ui(self):

        layout = [
            [sg.Graph(canvas_size=(551,551),
                      graph_bottom_left = (0,550),
                      graph_top_right = (550,0),
                      background_color='black',
                      key='graph',
                      drag_submits=True,
                      enable_events=True,
                      metadata=self)],
            [sg.Button('Approx Poly', key='approx_poly'),
             sg.Text('Epsilon'),
             sg.InputText('1.0',key='epsilon', size=(5,1))],
            [sg.Button('Ellipsify', key='ellipsify')]
        ]
        self.window = sg.Window('Ellipse Fitter', layout, finalize=True)
        self.render()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event in {'graph','graph+UP'}:
                self.handle_mouse(event, values)
            elif event == 'approx_poly':
                self.approx_poly(event, values)
            elif event == 'ellipsify':
                self.ellipsify(event, values)
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
