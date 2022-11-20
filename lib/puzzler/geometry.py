import puzzler

import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Line:

    pts: np.array

    def __repr__(self):
        x0, y0 = self.pts[0]
        x1, y1 = self.pts[1]
        return f"Line(({x0:.1f},{y0:.1f}),({x1:.1f},{y1:.1f}))"

@dataclass
class Ellipse:

    center: np.array
    semi_major: float
    semi_minor: float
    phi: float

    def __repr__(self):
        cx, cy = self.center
        maj, min = self.semi_major, self.semi_minor
        phi = self.phi
        return f"Ellipse(x,y={cx:.1f},{cy:.1f} {maj=:.1f} {min=:.1f} {phi=:.3f})"

def fit_ellipse_to_points(points):
    
    try:
        coeffs = fit_ellipse(points)
    except np.linalg.LinAlgError:
        return None

    # throw away complex results, wtf is going on here anyway?
    if coeffs.dtype.kind == 'c':
        return None

    cx, cy, major, minor, e, phi = cart_to_pol(coeffs)

    return Ellipse(np.array((cx, cy)), major, minor, phi)

class DistanceToEllipseComputer:

    def __init__(self, ellipse):
        
        self.center     = ellipse.center.copy()
        self.semi_major = ellipse.semi_major
        self.semi_minor = ellipse.semi_minor

        # rotate the point into the coordinates of an axis-aligned ellipse
        c, s = math.cos(-ellipse.phi), math.sin(-ellipse.phi)
        
        self.rot = np.array((( c, s), (-s, c)))
        
    def __call__(self, pts):

        pts_local   = (pts - self.center) @ self.rot
        pts_nearest = np_nearest_point_to_axis_aligned_ellipse_at_origin(self.semi_major, self.semi_minor, pts_local)
        return np.linalg.norm(pts_nearest - pts_local)

# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

def fit_ellipse(pts):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    x = pts[:,0]
    y = pts[:,1]
    if pts.dtype.kind != 'f':
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

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

def get_ellipse_points(ellipse, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0 = ellipse.center.tolist()
    ap, bp = ellipse.semi_major, ellipse.semi_minor
    phi    = ellipse.phi
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return np.vstack((x,y)).transpose()

def nearest_point_to_axis_aligned_ellipse_at_origin(semi_major, semi_minor, p):
    
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

    return (math.copysign(a * tx, p[0]), math.copysign(b * ty, p[1]))

def np_nearest_point_to_axis_aligned_ellipse_at_origin(semi_major, semi_minor, p):

    assert p.shape[-1] == 2
    p = np.atleast_2d(p)
    
    pxy = np.absolute(p)

    txy = np.full(p.shape, 0.707)

    a = semi_major
    b = semi_minor
    ab = np.array((a, b))

    for _ in range(0, 3):

        xy = txy * ab

        exy = np.array(((a*a - b*b) / a, (b*b - a*a) / b)) * (txy ** 3)

        rxy = xy - exy
        qxy = pxy - exy

        r = np.linalg.norm(rxy, axis=1)
        q = np.linalg.norm(qxy, axis=1)

        r_over_q = r / q
        txy = np.clip((qxy * r_over_q[:,np.newaxis] + exy) / ab, 0., 1.)
        txy = txy / np.linalg.norm(txy, axis=1)[:,np.newaxis]

    return np.copysign(txy * ab, p)
            
