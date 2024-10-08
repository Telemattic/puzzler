import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import cv2 as cv
import functools
import json
import math
import numpy as np
import re
import scipy.spatial.distance
import time
from pprint import *
from puzzbot.camera.calibrate import CornerDetector, BigHammerCalibrator, draw_detected_corners
import puzzler

class PieceFinder:

    def __init__(self):
        self.image_dpi = 600.
        self.image_binary_threshold = 130
        self.min_size_mm = 10
        self.max_size_mm = 50
        self.margin_px = 5

    def find_pieces_one_image(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, self.image_binary_threshold, 255, cv.THRESH_BINARY)[1]
            
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
        thresh = cv.erode(thresh, kernel)
        thresh = cv.dilate(thresh, kernel)

        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
        retval = []
    
        image_h, image_w = gray.shape
        margin_px = self.margin_px

        min_size_px = self.min_size_mm * self.image_dpi / 25.4
        max_size_px = self.max_size_mm * self.image_dpi / 25.4
    
        for c in contours:
            r = cv.boundingRect(c)
            x, y, w, h = r
            if min_size_px <= w <= max_size_px and min_size_px <= h <= max_size_px:
                if margin_px < x and x+w < image_w - margin_px and margin_px < y and y+h < image_h - margin_px:
                    retval.append(r)

        return retval

class ImageProjection:

    def __init__(self, coord_mm, px_per_mm):
        self.coord_mm = coord_mm
        self.px_per_mm = px_per_mm

    def px_to_mm(self, px):
        mm_per_px = 1 / self.px_per_mm
        return np.asarray(px) * (mm_per_px, -mm_per_px) + self.coord_mm

    def mm_to_px(self, mm):
        return (np.asarray(mm) - self.coord_mm) * (self.px_per_mm, -self.px_per_mm)

def find_pieces(scanpath, crop=None, alpha=0., beta=0.):

    finder = PieceFinder()

    R = np.array([[np.cos(alpha), -np.sin(beta)],
                  [np.sin(alpha), np.cos(beta)]])

    images = []
    for i in os.scandir(scanpath):
        if m := re.match('scan_(\d+)_(\d+)\.png', i.name):
            x, y = int(m[1]), int(m[2])
            print(f"{i.path=} {x=} {y=}")
            img = cv.imread(i.path)
            if crop is not None:
                x1, y1, x2, y2 = crop
                y1 += 100
                y2 -= 100
                img = img[y1:y2,x1:x2].copy()
            xy = R @ np.array((x,y)).T
            images.append((img, xy))

    dpi = 600.
    px_per_mm = dpi / 25.4

    all_pieces = []
    for image_no, (image, xy_mm) in enumerate(images):

        pieces = finder.find_pieces_one_image(image)

        print(f"{image_no=} found {len(pieces)} pieces")
        with np.printoptions(precision=1):
            print(f"  {xy_mm=} {px_per_mm=:.1f}")
        
        proj = ImageProjection(xy_mm, px_per_mm)
        
        for r in pieces:
            x, y, w, h = r
            center_px = np.array((x + w/2, (y + h/2)))
            radius_px = math.hypot(w, h) / 2

            center_mm = proj.px_to_mm(center_px)
            radius_mm = radius_px / px_per_mm

            with np.printoptions(precision=1):
                print(f"  {center_px=} {radius_px=:.1f} {center_mm=} {radius_mm=:.1f}")

            all_pieces.append((center_mm, radius_mm))

    scale = .25

    images2 = []
    for image_no, (image, xy_mm) in enumerate(images):
        proj = ImageProjection(np.zeros(2), px_per_mm*scale)
        xy_px = proj.mm_to_px(xy_mm)
        images2.append((cv.resize(image, None, fx=scale, fy=scale), int(xy_px[0]), int(xy_px[1])))

    min_x = min(x for i, x, y in images2)
    max_x = max(x + i.shape[1] for i, x, y in images2)
    min_y = min(y for i, x, y in images2)
    max_y = max(y + i.shape[0] for i, x, y in images2)

    print(f"{min_x=} {max_x=} {min_y=} {max_y=}")
    pano = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint16)
    counts = np.zeros(pano.shape, dtype=np.uint8)
    print(f"{pano.shape=}")

    for img, x, y in images2:
        x -= min_x
        y -= min_y
        h, w = img.shape[:2]
        pano[y:y+h,x:x+w,:] += img
        counts[y:y+h,x:x+w,:] += 1

    pano = np.array(pano // np.maximum(counts,1), dtype=np.uint8)

    if True:
        # this projection will get us from mm to pixels in the
        # coordinate system of images[0], then we need to get to the
        # panorama's coordinate system, which means translating by the
        # placement of images2[0]
        proj = ImageProjection(np.zeros(2), scale * px_per_mm)
        offset_px = np.array((-min_x, -min_y))
        print(f"{offset_px=}")
        
        for center_mm, radius_mm in all_pieces:
            center_px = np.array(proj.mm_to_px(center_mm) + offset_px, dtype=np.int32)
            radius_px = int(radius_mm * proj.px_per_mm)
            # print(f"{center_px=} {radius_px=}")
            cv.circle(pano, center_px, radius_px, (0,255,255))

    dist = scipy.spatial.distance.pdist([i for i, _ in all_pieces])

    n = len(all_pieces)
    
    uniq_ids = dict((i,i) for i in range(n))

    print("Scanning for redundant pieces, {n} total pieces")
    k = 0
    for i in range(n-1):
        l = n-(i+1)
        overlaps = np.flatnonzero(dist[k:k+l] < all_pieces[i][1])+i+1
        if len(overlaps):
            print(f"piece {i} overlaps pieces {overlaps}")
            if uniq_ids[i] == i:
                for m in overlaps:
                    uniq_ids[m] = i
            else:
                # we expect all the overlapping pieces of piece 'i' to
                # also have overlapped whatever it is that 'i'
                # overlapped
                assert all(uniq_ids[m] == uniq_ids[i] for m in overlaps)
        k += l
    assert k == len(dist)

    keepers = []
    for k, v in uniq_ids.items():
        if k == v:
            keepers.append(all_pieces[k])

    print(f"Found {n} pieces with initial scan, reduced to {len(keepers)} unique pieces")

    if False:
        for img, x, y in images2:
            x -= min_x
            y -= min_y
            h, w = img.shape[:2]
            cv.rectangle(pano, (x, y, w, h), (255,0,0), thickness=2)

    opath = os.path.join(scanpath, "pano.png")
    cv.imwrite(opath, pano)

    to_scan = [k[0].tolist() for k in keepers]

    opath = os.path.join(scanpath, 'toscan.json')
    with open(opath, 'w') as f:
        f.write(json.dumps(to_scan, indent=4))
            
def find_pieces_in_image(image, proj):

    min_size_mm = 10
    max_size_mm = 60
    threshold   = 128

    min_area = (proj.px_per_mm * min_size_mm) ** 2
    max_area = (proj.px_per_mm * max_size_mm) ** 2

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)[1]
    
    contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    print(f"Found {len(contours)} contours!")

    retval = []
    for i, c in enumerate(contours):
        a = cv.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        
        x, y, w, h = cv.boundingRect(c)
        center_px = (x + w/2, y + h/2)
        radius_px = math.hypot(w,h) / 2
        center_mm = proj.px_to_mm(center_px)
        radius_mm = radius_px / proj.px_per_mm
        retval.append((center_mm, radius_mm, c))

    return retval

def make_pano(scanpath, blend=False, dpi=150.):

    images = []
    for i in os.scandir(scanpath):
        if m := re.match('scan_(\d+)_(\d+)\.(jpg|png)', i.name):
            x, y = int(m[1]), int(m[2])
            print(f"{i.path=} {x=} {y=}")
            img = cv.imread(i.path)
            images.append((img, np.array((x,y))))

    dpi = 150.
    blend = True

    #
    
    pano_ul_mm = np.array((min(xy[0] for _, xy in images),
                           max(xy[1] for _, xy in images)))
    px_per_mm = dpi / 25.4

    proj = ImageProjection(pano_ul_mm, px_per_mm)
        
    images_px = []
    for i, xy_mm in images:
        xy_px = np.array(proj.mm_to_px(xy_mm), np.int32)
        images_px.append((i, xy_px))

    min_x = min(xy[0] for i, xy in images_px)
    max_x = max(xy[0] + i.shape[1] for i, xy in images_px)
    min_y = min(xy[1] for i, xy in images_px)
    max_y = max(xy[1] + i.shape[0] for i, xy in images_px)

    print(f"{min_x=} {max_x=} {min_y=} {max_y=}")

    if blend:
        pano = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint16)
        counts = np.zeros(pano.shape, dtype=np.uint8)
    else:
        pano = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint8)
        
    print(f"{pano.shape=}")

    for img, (x, y) in images_px:
        x -= min_x
        y -= min_y
        h, w = img.shape[:2]
        if blend:
            pano[y:y+h,x:x+w,:] += img
            counts[y:y+h,x:x+w,:] += 1
        else:
            pano[y:y+h,x:x+w,:] = img

    if blend:
        pano = np.array(pano // np.maximum(counts,1), dtype=np.uint8)
        
    cv.imwrite('pano.jpg', pano)

def stitch_pano(path):

    images = []
    for i in os.scandir(path):
        if re.match('scan_.*\.(jpg|png)', i.name):
            print(i.path)
            img = cv.imread(i.path)
            images.append(img)

    stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)

    status, pano = stitcher.stitch(images)

    if status != cv.Stitcher_OK:
        print(f"Stitching failed, {status=}")
    else:
        print(f"Success, {pano.shape=}")
        cv.imwrite('pano.jpg', pano)

def corners_to_dict(corners, ids):
    return dict(zip(ids, corners))

def global_calibrate(board, images):

    corners = {}
    
    for xy, image in images.items():
        print(f"{xy=}")
        the_corners, the_ids = CornerDetector(board).detect_corners(image)
        print(f"   {len(the_corners)} corners")
        corners[xy] = corners_to_dict(the_corners, the_ids)

    square_length = board.getSquareLength()
    board_n_cols, board_n_rows = board.getChessboardSize()

    gamma = (25.4 / 600) # mm/pixel

    print(f"{square_length=} {board_n_cols=} {board_n_rows=}")

    n_rows = 2 * sum(len(v) for v in corners.values())
    n_cols = 4

    A = np.zeros((n_rows, n_cols))
    b = np.zeros((n_rows,))

    r = 0
    for (xm, ym), image_corners in corners.items():
        for (ii, jj), (u, v) in image_corners.items():
            xc = ii * square_length
            yc = (board_n_rows - 1 - jj) * square_length
            A[r,1] = -ym
            A[r,2] = 1
            b[r] = xc - xm - gamma * u
            r += 1
            A[r,0] = xm
            A[r,3] = 1
            b[r] = yc - ym + gamma * v
            r += 1

    # solve Ax = b
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"{x=}")
    alpha, beta = x[0], x[1]
    error = alpha - beta
    for units, scale in [('radians ', 1.), ('degrees ', 180./math.pi), ('mm/meter', 1000.)]:
        print(f"{units}: alpha={alpha*scale:7.3f} beta={beta*scale:7.3f} error={error*scale:7.3f}")

def maximum_calibrated_rect(image, board):

    corners, ij_pairs = CornerDetector(board).detect_corners(image)

    start = time.perf_counter()

    rect = BigHammerCalibrator(board).maximum_rect(ij_pairs)

    elapsed = time.perf_counter() - start
    
    print(f"{rect=} {elapsed=:.3f}")
    return rect

def get_corner_dict_for_image(path, board):
    img = cv.imread(path)
    corners, ids = CornerDetector(board).detect_corners(img)
    return dict(zip(ids, corners))

def repeat_alignment_test(path1, path2, board):

    corners0 = get_corner_dict_for_image(path1, board)
    print(f"{path1}: {len(corners0)} corners")

    corners1 = get_corner_dict_for_image(path2, board)
    print(f"{path2}: {len(corners1)} corners")

    overlap = list(set(corners0) & set(corners1))
    print(f"{len(overlap)} shared corners")

    n_rows = len(overlap)*2
    A = np.zeros((n_rows,3))
    b = np.zeros((n_rows,))

    r = 0
    for ij in overlap:
        x0, y0 = corners0[ij]
        x1, y1 = corners1[ij]
        A[r,0] = -y1
        A[r,1] = 1
        b[r] = x0 - x1
        r += 1
        A[r,0] = x1
        A[r,2] = 1
        b[r] = y0 - y1
        r += 1

    theta, dx, dy = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"{theta=} ({theta*180/math.pi:.3f} deg) {dx=} ({dx*25.4/600.:.3f} mm) {dy=} ({dy*25.4/600.:.3f} mm)")
    print()
    
def repeat_alignment_directories(dir1, dir2, board):

    for i in os.scandir(dir1):
        if i.is_file() and i.name.endswith('.png'):
            path1 = i.path
            path2 = os.path.join(dir2, i.name)
            repeat_alignment_test(path1, path2, board)

def make_corners_detectable(path, board):

    def count_corners(img, median_ksize = None):
        detector = CornerDetector(board)
        detector.median_ksize = median_ksize
        corners, ids = detector.detect_corners(img)
        if len(corners):
            size = (12 * img.shape[1]) // 4032
            draw_detected_corners(img, corners, size=size)
        return 0 if corners is None else len(corners)

    print(path)
    
    img = cv.imread(path)
    
    print(f"  initially {count_corners(img)} corners detected")

    img2 = img.copy()
    print(f"  median_prefilter: {count_corners(img2, 5)} corners detected")
    cv.imwrite('test_d/median_prefilter.png', img2)

    for ksize in (3, 5, 7):

        img2 = cv.medianBlur(img, ksize)
        print(f"  medianBlur({ksize}): {count_corners(img2)} corners detected")
        cv.imwrite(f'test_d/median_blur_{ksize}.png', img2)

    for sigma in (3, 5, 7):

        img2 = cv.GaussianBlur(img, None, sigma)
        print(f"  GaussianBlur({sigma}): {count_corners(img2)} corners detected")
        cv.imwrite(f'test_d/gaussian_blur_{sigma}.png', img2)

    for scale in (2, 3, 4, 5):

        img2 = cv.resize(img, None, fx=1./scale, fy=1./scale)
        print(f"  resize(1/{scale}): {count_corners(img2)} corners detected")
        cv.imwrite(f'test_d/resize_1_{scale}.png', img2)

    for ksize in (3, 5, 7):

        img2 = cv.blur(img, (ksize,ksize))
        print(f"  blur({ksize},{ksize}): {count_corners(img2)} corners detected")
        cv.imwrite(f'test_d/blur_{ksize}_{ksize}.png', img2)

    for d in (5, 7, 9):
        for sigma in 20, 50, 100, 150, 200:
            continue
            img2 = cv.bilateralFilter(img, d, sigma, sigma)
            print(f"  bilateralFilter({d=}, {sigma=}): {count_corners(img2)} corners detected")
            cv.imwrite(f'test_d/bilateral_d{d}_sigma{sigma}.png', img2)

def compute_hu_moments_for_pieces(pieces):
    def moments(piece):
        return cv.HuMoments(cv.moments(piece.points))
    return dict((i.label, moments(i)) for i in pieces)

def compare_moments(path_a, path_b):

    puzzle_a = puzzler.file.load(path_a)
    puzzle_b = puzzler.file.load(path_b)

    # dict_a = compute_hu_moments_for_pieces(puzzle_a.pieces)
    # dict_b = compute_hu_moments_for_pieces(puzzle_b.pieces)

    summary = []
    for piece_b in puzzle_b.pieces:

        best = []
        for mode in (cv.CONTOURS_MATCH_I1, cv.CONTOURS_MATCH_I2, cv.CONTOURS_MATCH_I3):

            scores = []
            for piece_a in puzzle_a.pieces:
                score = cv.matchShapes(piece_a.points, piece_b.points, mode, 0.)
                scores.append((score, piece_a.label))

            scores.sort()
            print(f"piece_b={piece_b.label}, {mode=} scores:")
            for i, (score, label) in enumerate(scores):
                print(f"  {i} {score:.6f} {label}")
            print()

            best.append(scores[0][1])

        summary.append((piece_b.label, best))

    print("SUMMARY")
    for b_label, a_labels in summary:
        print(f"{b_label}:", ', '.join(a_labels))

def find_pieces_in_pano(ipath, opath):

    image = cv.imread(ipath)
    scale = .5
    image = cv.resize(image, None, fx=scale, fy=scale)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)[1]
    
    contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    print(f"Found {len(contours)} contours!")

    scale2 = scale * scale
    min_size = 3600 * scale2
    little_guy = 20_000 * scale2
    big_guy = 50_000 * scale2

    keep = []
    for i, c in enumerate(contours):
        a = cv.contourArea(c)
        if a < min_size:
            continue
        keep.append(c)
        print(f"contour[{i}]: length={len(c)} area={a}")
        
    print(f"Keeping {len(keep)} contours")

    for c in keep:
        a = cv.contourArea(c)
        color = (0, 0, 255)
        if a < little_guy:
            color = (0, 255, 255)
        if a > big_guy:
            color = (255, 255, 0)
        cv.drawContours(image, [c], -1, color, 2)

    cv.imwrite('pano-gray.png', gray)
    cv.imwrite(opath, image)

def find_pieces_in_pano2(ipath, opath):

    image = cv.imread(ipath)
    scale = .5
    image = cv.resize(image, None, fx=scale, fy=scale)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)[1]
    
    contours, links = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    links = np.squeeze(links)
    print(f"Found {len(contours)} contours! {len(links)=}")
    pprint(links)

    # hierarchy "record" is [next,prev,first_child,parent], -1 is a sentinel value for None
    # next: next child at same level of hierarchy with same parent
    # prev: prev child at same level of hierarchy with same parent
    # first_child: index of first child of this contour (i.e. child with prev==-1)
    # parent: index of parent contour
    parents = links[:,3]

    scale2 = scale * scale
    min_size = 3600 * scale2
    little_guy = 20_000 * scale2
    big_guy = 50_000 * scale2

    keep = []
    for i, c in enumerate(contours):
        a = cv.contourArea(c)
        if a < min_size:
            continue
        keep.append(i)
        print(f"contour[{i}]: length={len(c)} area={a}")

    print(f"Keeping {len(keep)} contours")

    nodes = set()
    edges = []
    for i in keep:
        if i in nodes:
            continue
        nodes.add(i)
        p = parents[i]
        while p != -1:
            if p != 1:
                edges.append((i, p))
            if p in nodes:
                break
            nodes.add(p)
            i = p
            p = parents[i]

    with open('pano-graph.dot', 'w') as f:
        print("digraph G {", file=f)
        for i, j in edges:
            print(f"  {j} -> {i}", file=f)
        print("}", file=f)
            
    for i in keep:
        c = contours[i]
        a = cv.contourArea(c)
        color = (0, 0, 255)
        if a < little_guy:
            color = (0, 255, 255)
        if a > big_guy:
            color = (255, 255, 0)
        cv.drawContours(image, [c], -1, color, 2)

    cv.imwrite('pano-gray.png', gray)
    cv.imwrite(opath, image)

def do_tsp():

    from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
    from python_tsp.utils import compute_permutation_distance

    image = cv.imread('pano.jpg')
    
    with open('centers.json') as f:
        nodes = json.load(f)
    vertexes = np.array([xy for xy, _ in nodes])

    print(f"{vertexes.shape=}")

    distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vertexes))

    distance_matrix[:,0] = 0

    naive_path = list(range(len(vertexes)))
    naive_distance = compute_permutation_distance(distance_matrix, naive_path)

    print(f"{naive_distance=}")

    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

    print(f"{distance=}")

    permutation2, distance2 = solve_tsp_local_search(
        distance_matrix, x0=permutation, perturbation_scheme='ps3')

    print(f"{distance2=}")

    vertexes = vertexes.astype(np.int32)

    cv.polylines(image, [vertexes], False, (255, 0, 0))
    cv.polylines(image, [vertexes[permutation]], False, (0, 0, 180), thickness=5)
    cv.polylines(image, [vertexes[permutation2]], False, (0, 255, 0), thickness=2)

    cv.imwrite('pano-tsp.jpg', image)

def main():

    if True:
        compare_moments('../100.json', 'temp-puzzle.json')
        return None

    if False:
        # scan_q: crop=None, alpha=.00429247, beta=.00293304
        # scan_x: crop=(741, 245, 3716, 2653) alpha=-.0369759 beta=-0.0371328
        find_pieces('scan_q', crop=None, alpha=.00429247, beta=.00293304)
        return None

    if True:
        do_tsp()
        return None

    if False:
        find_pieces_in_pano2('pano.jpg', 'pano-contours.jpg')
        return None
    
    if True:
        make_pano('.')
        return None

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    board = cv.aruco.CharucoBoard((33,44), 6., 3., aruco_dict)

    if False:
        make_corners_detectable('test_d/img_0_1.png', board)
        return None

    if False:
        repeat_alignment_directories('test_a', 'test_d', board)
        return None

    images = dict()
    for i in os.scandir('scan_x'):
        if m := re.match('img_(\d+)_(\d+)\.png', i.name):
            x = int(m[1])
            y = int(m[2])
            print(f"load {i.path}, {x=} {y=}")
            images[(x,y)] = cv.imread(i.path)

    print("global_calibrate!")
    global_calibrate(board, images)
    return None

if __name__ == '__main__':
    main()
