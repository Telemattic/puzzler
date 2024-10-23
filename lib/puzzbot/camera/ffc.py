import cv2 as cv
import functools
import numpy as np
import numpy.polynomial.polynomial as poly

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def compute_coefficients(img):
    step = 32
    kx = ky = 2
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv.GaussianBlur(gray, (31,31), 0)

    gray = gray / np.mean(gray)

    h, w = gray.shape
    s0 = step // 2
    x = np.arange(s0, w, step) * (1/w)
    y = np.arange(s0, h, step) * (1/h)
    z = gray[s0:h:step,s0:w:step]

    ret = polyfit2d(x, y, z, kx=kx, ky=ky, order=None)
    return ret[0].reshape((kx+1, ky+1))

class FlatFieldCorrector:

    def __init__(self, coeffs):
        self.coeffs = coeffs

    @functools.lru_cache(maxsize=2)
    def get_gain(self, w, h):
        x = np.arange(w) * (1/w)
        y = np.arange(h) * (1/h)
        fit = poly.polygrid2d(y, x, self.coeffs)
        return np.reciprocal(fit)

    def __call__(self, image):
        h, w = image.shape[:2]
        g = self.get_gain(w, h)
        if image.ndim == 3:
            g = g[:,:,np.newaxis]
        return np.uint8(np.clip(image * g, a_min=0, a_max=255))

