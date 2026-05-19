# cd ctools && python setup.py build_ext --debug
#
# copy *.pyd and *.pdb
#  from  ctools/build/lib.win-amd64-cpython-313
#  to    env/Lib/site-packages/puzzler

import os
import sys

lib = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, lib)

import numpy as np
import puzzler
import puzzbin
import scipy
import PIL
import palettable.tableau

def samples_for_bbox(bbox):
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    return np.swapaxes(np.indices((w,h)), 0, 2) + bbox[0]

def kdtree_compute(bbox, points):
    kdtree = scipy.spatial.KDTree(points)
    samples = samples_for_bbox(bbox)
    return kdtree.query(samples)

def cnpi_compute(bbox, points):
    return puzzbin.compute_nearest_point_image(bbox, points)

def distance_map_to_image(dist, max=None):

    assert 0 <= np.min(dist)
    if max is None:
        max = np.max(dist)
    return PIL.Image.fromarray(np.array(dist * (255./max), dtype=np.uint8))

def bbox_compute(points):
    return np.array((np.min(points,axis=0), np.max(points,axis=0)+1))

def validator(bbox, points):

    kd, ki = kdtree_compute(bbox, points)
    dist, indices = cnpi_compute(bbox, points)

    samples = samples_for_bbox(bbox)
    dist2 = np.linalg.norm(points[indices] - samples, axis=2)

    # print(dist)
    # print(dist2)

    assert np.allclose(dist, dist2)
    
    indices_mismatch = np.not_equal(ki,indices)
    PIL.Image.fromarray((indices_mismatch*255).astype(np.uint8)).save('indices_mismatch.png')

    indices_mismatch_and_dist_error = indices_mismatch & np.logical_not(np.isclose(kd,dist))
    PIL.Image.fromarray((indices_mismatch_and_dist_error*255).astype(np.uint8)).save('indices_mismatch_and_dist_error.png')

    assert np.all(np.isclose(kd, dist) | np.not_equal(ki,indices))
    
    max = np.max((np.max(kd), np.max(dist)))
    distance_map_to_image(kd, max).save('dist_kd.png')
    distance_map_to_image(dist, max).save('dist_cnpi.png')

    for i, s in enumerate(samples[indices_mismatch_and_dist_error]):
        x,y = tuple(s)
        print(f"error({i}) at {s}")
        print(f"  expected: index={ki[y,x]:d} point={points[ki[y,x]]} dist={kd[y,x]:.3f}")
        print(f"  actual:   index={indices[y,x]:d} point={points[indices[y,x]]} dist={dist[y,x]:.3f}")
        if i == 9:
            break;

    isclose = np.isclose(kd, dist)
    n_total = np.prod(kd.shape)
    n_close = np.sum(isclose)
    print(f"{n_total=} {n_close=}, {n_total-n_close} not close")
    assert np.all(isclose)

    assert np.all(0 <= indices) and np.all(indices < len(points));

def cut_window(bbox, points):

    (x0, y0), (x1, y1) = bbox

    subset = points[(x0 <= points[:,0]) & (points[:,0] < x1) & (y0 <= points[:,1]) & (points[:,1] < y1)] - (x0,y0)
    subset = subset[np.lexsort((subset[:,1], subset[:,0]))]
    return ((0, 0), (x1-x0,y1-y0)), subset.astype(np.int32)

def index_difference_image(src):
    n = np.max(src)+1

    def compute_diff(x,y):

        diff = 0
        if 0 < x:
            d = src[x,y] - src[x-1,y]
            d = min(d % n, -d % n)
            if diff < d:
                diff = d
        if x+1 < w:
            d = src[x,y] - src[x+1,y]
            d = min(d % n, -d % n)
            if diff < d:
                diff = d
        if 0 < y:
            d = src[x,y] - src[x,y-1]
            d = min(d % n, -d % n)
            if diff < d:
                diff = d
        if y+1 < h:
            d = src[x,y] - src[x,y+1]
            d = min(d % n, -d % n)
            if diff < d:
                diff = d
        return d

    w, h = src.shape
    dst = np.zeros(src.shape,dtype=np.int32)
    for y in range(h):
        for x in range(w):
            dst[x,y] = compute_diff(x,y)

    return dst

def bitlength_image(src):
    return np.ceil(np.log2(src+1)).astype(np.uint8)

bbox = np.array(((-1,-1),(8,6)), dtype=np.int32)
points = np.array([(2,1),(1,3),(3,3)], dtype=np.int32)
np.set_printoptions(precision=3)

puzzle = puzzler.file.load('100.json')
A1 = puzzle.pieces[0].points
A2 = A1 - np.min(A1, axis=0)
A3 = np.unique(np.array(A2 / 16, np.uint8), axis=0)
A4 = A3[(A3[:,0] < 8) & (A3[:,1] < 8)]
A4_bbox = bbox_compute(A4)

# colors = [tuple(c/255 for c in color) for color in palettable.tableau.Tableau_20.colors]
# indices = cnpi_compute(bbox_compute(A1),A1)[1]
# print(len(colors))

