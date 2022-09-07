import cv2
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


class EllipseFitter:

    def __init__(self, filename):
        self.filename = filename
        self.ellipse_pts = None
        self.mouse_rect = None
        self.approx_pts = None
        self.perimeter_pts = None

    def fit_ellipse_to_rect(self, rect):

        print(f"fit_ellipse_to_rect: {rect=}")

        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)

        x0, y0 = rect[0]
        x1, y1 = rect[1]
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1

        image_pts = np.nonzero(img[y0:y1, x0:x1])

        coeffs = fit_ellipse(image_pts[1], image_pts[0])

        ellipse_pts = get_ellipse_pts(cart_to_pol(coeffs))

        ret = list(zip(ellipse_pts[0]+x0, ellipse_pts[1]+y0))

        return ret

    def approx_poly(self, event, values):

        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 2
        c = max(contours[0], key=cv2.contourArea)

        self.perimeter_pts = list(tuple(i[0]) for i in c)

        eps = float(self.window['epsilon'].get())

        approx_pts = cv2.approxPolyDP(c, eps, True)
        # print(f"{len(approx_pts)=} {approx_pts=}")

        self.approx_pts = approx_pts

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

        if self.approx_pts is not None and False:
            xy_tuples = list(tuple(i[0]) for i in self.approx_pts)
            graph.draw_lines(xy_tuples, color='#00ff00', width=2)

            for i in range(len(xy_tuples)):
                x0, y0 = xy_tuples[i-2]
                x1, y1 = xy_tuples[i-1]
                x2, y2 = xy_tuples[i]
                area = (x1-x0) * (y2-y1) - (x2-x1)*(y1-y0)
                color = 'red' if area >= 0 else 'blue'
                graph.draw_point((x1,y1), size=5, color=color)

        if self.ellipse_pts:
            graph.draw_lines(self.ellipse_pts, color='blue', width=2)

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
            [sg.Button('Approx Poly', key='approx_poly'), sg.Text('Epsilon'), sg.InputText('1.0',key='epsilon', size=(5,1))]
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
            else:
                print(event, values)

def main():

    e = EllipseFitter('A02_extract.png')
    e.ui()

if __name__ == '__main__':
    main()
