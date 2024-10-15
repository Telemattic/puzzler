import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import cv2 as cv
import math
import numpy as np

from puzzbot.camera.calibrate import CornerDetector

def get_corners(board, image):
    cd = CornerDetector(board)
    corners, ids = cd.detect_corners(image)
    return dict(zip(ids,corners))

def fcal(board, initial_theta, image0, image1):

    corners0 = get_corners(board, image0)
    corners1 = get_corners(board, image1)

    overlap = set(corners0) & set(corners1)

    print(f"{len(overlap)} common corners")

    n_rows = 2 * len(overlap)
    n_cols = 3

    theta = initial_theta
    for i in range(5):

        theta_deg = theta * 180. / math.pi
        print(f"iteration {i}, {theta=:.5f} rad ({theta_deg:.2f} deg)")

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows,))

        c = math.cos(theta)
        s = math.sin(theta)

        r = 0
        for i in overlap:
            u0, v0 = corners0[i]
            u1, v1 = corners1[i]

            A[r][0] = -u0*s - v0*c
            A[r][1] = 1
            b[r] = u1 - u0*c + v0*s
            r += 1

            A[r][0] = u0*c - v0*s
            A[r][2] = 1
            b[r] = v1 - u0*s - v0*c
            r += 1

        # print("A=", A)
        # print("b=", b)

        # solve Ax = b
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        alpha, x, y = x
        alpha_deg = alpha * 180. / math.pi
        print(f"{alpha=:.5f} rad ({alpha_deg:.2f} deg), {x=:.1f} px, {y=:.1f} px")

        c = math.cos(theta + alpha)
        s = math.sin(theta + alpha)
        
        denom = (1-c)**2 + s**2
        u_center = (x*(1-c) - y*s) / denom
        v_center = (x*s + y*(1-c)) / denom

        print(f"center: ({u_center:.3f},{v_center:.3f})")

        if False:
            m = np.array([(c, -s, x),
                          (s, c, y),
                          (0, 0, 1)])
            p = np.array([(u_center,),
                          (v_center,),
                          (1,)])

            with np.printoptions(precision=3):
                print(f"{m=}")
                print(f"{p=}")
                print(f"{m @ p=}")

        theta += alpha
        print()

        if math.fabs(alpha) < .001:
            break

def main():

    board = cv.aruco.CharucoBoard(
        (8, 8), 6., 3., cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100))

    print('-' * 80)
    for steps in range(0,51,5):
        path0 = f'finger_{steps}_B.jpg'
        path1 = f'finger_{steps}_C.jpg'
        print(f"{steps=} {path0=}, {path1=}\n")
        image0 = cv.imread(path0, cv.IMREAD_COLOR)
        image1 = cv.imread(path1, cv.IMREAD_COLOR)
        fcal(board, steps * (math.pi/100), image0, image1)
        print('-' * 80)

if __name__ == '__main__':
    main()
