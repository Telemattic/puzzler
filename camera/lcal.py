import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import cv2 as cv
import math
import numpy as np
import re
from matplotlib.pylab import cm

def lighting_calibrate(ipath, opath):

    img = cv.imread(ipath, cv.IMREAD_COLOR)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (17,17), 0)

    print(f"{np.min(gray)=} {np.max(gray)=}")

    for threshold in range(8,48,8):
        tmp = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)[1]
        contours = cv.findContours(tmp, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
        cv.drawContours(img, contours, -1, (0,127,0))

    if True:
        cv.imwrite(opath, img)
    else:
        heatmap = cv.cvtColor(np.uint8(cm.afmhot(gray)*255), cv.COLOR_RGBA2BGR)
        cv.imwrite(opath, heatmap)
        
def main():

    lighting_calibrate('pano_397_1107.jpg', 'lightmap.png')

if __name__ == '__main__':
    main()
