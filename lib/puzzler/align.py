import puzzler
import cachetools
import math
import numpy as np
import cv2 as cv
import scipy
from dataclasses import dataclass, field

class Coord:

    def __init__(self, angle=0., xy=(0.,0.)):
        self._angle = angle
        self._xy = np.array(xy, dtype=np.float64)
        self._xform = None

    def from_matrix(m):
        angle = math.atan2(m[1,0], m[0,0])
        x, y = m[0,2], m[1,2]
        return Coord(angle, (x,y))

    def compose(c1, c2):
        angle = math.fmod(c1.angle + c2.angle, 2. * math.pi)
        # if angle >= math.pi:
        #     angle -= 2. * math.pi
        return Coord(angle, c1.xform.apply_v2(c2.xy))
    
    def __repr__(self):
        return f"Coord({self._angle!r}, {self._xy!r})"

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, v):
        self._angle = v
        self._xform = None

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, v):
        self._xy = np.array(v, dtype=np.float64)
        self._xform = None

    @property
    def xform(self):
        if self._xform is None:
            self._xform = (puzzler.render.Transform()
                            .translate(self._xy)
                            .rotate(self._angle))
        return self._xform

    @property
    def matrix(self):
        return self.xform.matrix

    def copy(self):
        return Coord(self._angle, tuple(self._xy))

def ring_slice(data, a, b):
    return np.concatenate((data[a:], data[:b])) if a >= b else data[a:b]

def compute_rigid_transform(P, Q):

    m = P.shape[0]
    assert P.shape == Q.shape == (m, 2)

    # want the optimized rotation and translation to map P -> Q

    Px, Py = np.sum(P, axis=0)
    Qx, Qy = np.sum(Q, axis=0)

    A00 =np.sum(np.square(P))
    A = np.array([[A00, -Py, Px], 
                  [-Py,  m , 0.],
                  [ Px,  0., m ]], dtype=np.float64)

    u0 = np.sum(P[:,0]*Q[:,1]) - np.sum(P[:,1]*Q[:,0])
    u = np.array([u0, Qx-Px, Qy-Py], dtype=np.float64)

    return np.linalg.lstsq(A, u, rcond=None)[0]

class DistanceImage:

    # try to keep this from running away
    cache = cachetools.LRUCache(maxsize=1026)

    @staticmethod
    def Factory(piece):
        o = DistanceImage.cache.get(piece.label)
        if o is None:
            o = DistanceImage(piece)
            DistanceImage.cache[piece.label] = o

        return o

    def __init__(self, piece, compress=True):

        pp = piece.points
        
        self.ll = np.min(pp, axis=0) - 256
        self.ur = np.max(pp, axis=0) + 256
        self.compress = compress

        w, h = self.ur + 1 - self.ll
        cols = pp[:,0] - self.ll[0]
        rows = pp[:,1] - self.ll[1]

        piece_image = np.ones((h, w), dtype=np.uint8)
        piece_image[rows, cols] = 0

        dist_image = cv.distanceTransform(piece_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        # pixels interior to the piece are 0, exterior are 1
        is_exterior_mask = cv.floodFill(piece_image, None, -self.ll, 0)[1]

        # positive distance is distance from the piece (external),
        # negative distance is distance from the boundary of the piece
        # for internal points
        signed_dist_image = np.where(is_exterior_mask, dist_image, -dist_image)

        if self.compress:
            signed_dist_image = np.array(signed_dist_image * 16, dtype=np.int16)

        self.dist_image = signed_dist_image

    def query(self, points):

        # could consider scipy.interpolate.RegularGridInterpolator,
        # either with nearest or linear interpolation
        h, w = self.dist_image.shape
        # n = len(points)
        
        # rows = np.empty(n, dtype=np.int32)
        # np.clip(points[:,1] - self.ll[1], a_min=0, a_max=h-1, out=rows, casting='unsafe')
        
        # cols = np.empty(n, dtype=np.int32)
        # np.clip(points[:,0] - self.ll[0], a_min=0, a_max=w-1, out=cols, casting='unsafe')

        rows = np.int32(np.clip(points[:,1] - self.ll[1], a_min=0, a_max=h-1))
        cols = np.int32(np.clip(points[:,0] - self.ll[0], a_min=0, a_max=w-1))

        ret = self.dist_image[rows, cols]
        return ret * (1/16) if self.compress else ret

class NormalsComputer:

    def __init__(self, baseline = 10):
        self.baseline = baseline

    def __call__(self, points, indices):

        n = len(points)

        p0 = points[(indices - self.baseline) % n]
        p1 = points[(indices + self.baseline) % n]
        v = p1 - p0
        l = np.linalg.norm(v, axis=1)
        # l can be zero when the perimeter is degenerate and the same
        # (x,y) appears multiple times in it, e.g. see tab 3 of piece
        # N34 from the 1000-piece puzzle
        l = np.where(l > 0., l, 1.)
        v = v / l[:,np.newaxis]

        normals = np.array((-v[:,1], v[:,0])).T
        
        return normals

class DistanceQueryCache:

    def __init__(self, purge_interval=None):
        self.serial_no = 1
        self.cache = dict()
        self.stats = {'n_read':0, 'n_write':0, 'read_miss':0, 'read_hit':0, 'n_purge':0, 'n_bytes':0}
        self.purge_interval = purge_interval
        self.next_purge = purge_interval

    def query(self, dst_piece, dst_coords, src_piece, src_coords):

        key = self.make_key(dst_piece, dst_coords, src_piece, src_coords)
        if (retval := self.read(key)) is not None:
            return retval

        transform = puzzler.render.Transform()
        transform.rotate(-dst_coords.angle).translate(-dst_coords.xy)
        transform.translate(src_coords.xy).rotate(src_coords.angle)

        di = DistanceImage.Factory(dst_piece)
        retval = di.query(transform.apply_v2(src_piece.points))
        retval.setflags(write=False)

        self.write(key, retval)

        return retval
        
    def make_key(self, dst_piece, dst_coords, src_piece, src_coords):
        def coords_key(c):
            return (c.angle, c.xy[0], c.xy[1])
        return (dst_piece.label, coords_key(dst_coords),
                src_piece.label, coords_key(src_coords))

    def read(self, key):
        
        if self.next_purge is not None:
            if self.next_purge <= self.stats['n_read']:
                self.purge()
            while self.next_purge <= self.stats['n_read']:
                self.next_purge += self.purge_interval
            
        self.stats['n_read'] += 1
        entry = self.cache.get(key)
        if not entry:
            self.stats['read_miss'] += 1
            return None
        
        entry[1] = self.serial_no
        self.stats['read_hit'] += 1
        return entry[0]

    def write(self, key, value):
        self.stats['n_write'] += 1
        self.stats['n_bytes'] += value.nbytes
        self.cache[key] = [value, self.serial_no]

    def purge(self):
        old_keys = [k for k, v in self.cache.items() if v[1] < self.serial_no]
        for k in old_keys:
            v = self.cache.pop(k)
            self.stats['n_bytes'] -= v[0].nbytes
        self.serial_no += 1
        self.stats['n_purge'] += len(old_keys)
