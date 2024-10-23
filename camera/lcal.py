import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import cv2 as cv
import functools
import json
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib.pylab import cm

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

def compute_coefficients(img, degree=2):
    step = 32
    kx = ky = degree
    
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

def lighting_calibrate(ipath, opath):

    img = cv.imread(ipath, cv.IMREAD_COLOR)
    #    img = cv.resize(img, None, fx=.25, fy=.25)

    coeffs = compute_coefficients(img, 4)
    img2 = FlatFieldCorrector(coeffs)(img)

    o = {'coeffs': coeffs.tolist()}
    print(json.dumps(o, indent=4))

    cv.imwrite('uncorrected.jpg', img)
    cv.imwrite('corrected.jpg', img2)

def lighting_calibrate2(ipath, opath):

    kx = ky = 3

    img = cv.imread(ipath, cv.IMREAD_COLOR)
    # img = cv.resize(img, None, fx=.25, fy=.25)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (127, 127), 0)

    # flat_field = cv.resize(gray, None, fx=.25, fy=.25)
    flat_field = gray
    flat_field = flat_field / np.mean(flat_field)
    print(f"{flat_field.dtype=}")

    h, w = flat_field.shape
    x = np.arange(16,w,32) / w
    y = np.arange(16,h,32) / h
    z = flat_field[16:h:32,16:w:32]

    ret = polyfit2d(x, y, z, kx=kx, ky=ky, order=None)

    coeffs = ret[0].reshape((kx+1,ky+1))
    print(f"{coeffs=}")

    print(f"{np.min(flat_field)=} {np.max(flat_field)=}")

    x = np.arange(w) / w
    y = np.arange(h) / h
    fitted_surf = poly.polygrid2d(y, x, coeffs)
    print(f"{np.min(fitted_surf)=} {np.max(fitted_surf)=}")

    gain = np.reciprocal(fitted_surf)
    print(f"{np.min(gain)=} {np.max(gain)=}")

    fitted_surf = (fitted_surf - np.min(fitted_surf)) / (np.max(fitted_surf) - np.min(fitted_surf))
    cv.imwrite('fitted_surf.png', cv.cvtColor(np.uint8(cm.jet(fitted_surf) * 255), cv.COLOR_RGBA2BGR))

    uncorrected = cv.resize(gray, (w,h))
    cv.imwrite('uncorrected.png', uncorrected)
    print(f"{np.min(uncorrected)=} {np.max(uncorrected)=}")

    corrected = np.uint8(uncorrected * gain)
    cv.imwrite('corrected.png', corrected)
    print(f"{np.min(corrected)=} {np.max(corrected)=}")

    def expand(gray, minval, maxval):
        tmp = (gray - minval) / (maxval - minval)
        return cv.cvtColor(np.uint8(cm.jet(tmp) * 255), cv.COLOR_RGBA2BGR)

    uncorrected_rgb = cv.resize(img, (w,h))
    corrected_rgb = np.uint8(uncorrected_rgb * gain[:,:,np.newaxis])

    uncorrected_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corrected_gray = np.uint8(uncorrected_gray * gain)
    
    minval = min(np.min(uncorrected_gray), np.min(corrected_gray))
    maxval = max(np.max(uncorrected_gray), np.max(corrected_gray))

    uncorrected_rgb = expand(uncorrected_gray, minval, maxval)
    corrected_rgb = expand(corrected_gray, minval, maxval)

    def to_rgb(color):
        rgba = np.uint8(255 * np.array(color)).tolist()
        return tuple(rgba[:3][::-1])

    uncorrected_gray = cv.GaussianBlur(uncorrected_gray, (127, 127), 0)
    corrected_gray = cv.GaussianBlur(corrected_gray, (127, 127), 0)
    
    for threshold in range(64,150,8):
        color = cm.afmhot(threshold)
        color2 = to_rgb(color)
        color2 = (192,0,192)

        if np.min(uncorrected_gray) <= threshold <= np.max(uncorrected_gray):
            tmp = cv.threshold(uncorrected_gray, threshold, 255, cv.THRESH_BINARY)[1]
            contours = cv.findContours(tmp, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
            cv.drawContours(uncorrected_rgb, contours, -1, color2, thickness=2, lineType=cv.LINE_AA)

        if np.min(corrected_gray) <= threshold <= np.max(corrected_gray):
            tmp = cv.threshold(corrected_gray, threshold, 255, cv.THRESH_BINARY)[1]
            contours = cv.findContours(tmp, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
            cv.drawContours(corrected_rgb, contours, -1, color2, thickness=2, lineType=cv.LINE_AA)

    cv.imwrite('uncorrected_rgb.png', uncorrected_rgb)
    cv.imwrite('corrected_rgb.png', corrected_rgb)
        
    print(f"{np.min(gray)=} {np.max(gray)=}")

    expanded = (gray - np.min(gray)) / (np.max(gray)-np.min(gray))
    img  = cv.cvtColor(np.uint8(cm.jet(expanded) * 255), cv.COLOR_RGBA2BGR)

    for threshold in range(96,150,8):
        tmp = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)[1]
        contours = cv.findContours(tmp, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
        cv.drawContours(img, contours, -1, (0,192,0))

    if True:
        cv.imwrite(opath, img)
    else:
        heatmap = cv.cvtColor(np.uint8(cm.afmhot(gray)*255), cv.COLOR_RGBA2BGR)
        cv.imwrite(opath, heatmap)
        
def main():

    # lighting_calibrate('pano_397_1107.png', 'lightmap.png')
    # lighting_calibrate('pano_black.png', 'lightmap.png')
    lighting_calibrate('pano_gray.png', 'lightmap.png')

if __name__ == '__main__':
    main()
