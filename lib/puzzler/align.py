import puzzler
import math
import numpy as np
import cv2 as cv
import scipy

class AffineTransform:

    def __init__(self, angle=0., xy=(0.,0.)):
        self.angle = angle
        self.dxdy  = np.array(xy, dtype=np.float64)

    def __repr__(self):
        return f"AffineTransform({self.angle!r}, {self.dxdy!r})"

    def invert_matrix(m):
        angle = math.atan2(m[1,0], m[0,0])
        x, y = m[0,2], m[1,2]
        return AffineTransform(angle, (x,y))

    def get_transform(self):
        return (puzzler.render.Transform()
                .translate(self.dxdy)
                .rotate(self.angle))

    def rot_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array(((c, -s),
                         (s,  c)))

    def copy(self):
        return AffineTransform(self.angle, tuple(self.dxdy))

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

    cache = dict()

    @staticmethod
    def Factory(piece):
        o = DistanceImage.cache.get(piece.label)
        if o is None:
            o = DistanceImage(piece)
            DistanceImage.cache[piece.label] = o

        return o

    def __init__(self, piece):

        pp = piece.points
        
        self.ll = np.min(pp, axis=0) - 256
        self.ur = np.max(pp, axis=0) + 256

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

        return self.dist_image[rows, cols]

class TabAligner:

    def __init__(self, dst):
        self.dst = dst
        self.kdtree = None
        self.distance_image = DistanceImage.Factory(dst)
        
    def compute_alignment(self, dst_tab_no, src, src_tab_no, refine=0):

        src_points = self.get_tab_points(src, src_tab_no)
        dst_points = self.get_tab_points(self.dst, dst_tab_no)

        dst_vec = dst_points[0] - dst_points[1] + dst_points[2] - dst_points[1]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[0] - src_points[1] + src_points[2] - src_points[1]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = AffineTransform(dst_angle-src_angle).get_transform().apply_v2(src_points)

        r, x, y = compute_rigid_transform(src_points_rotated, dst_points)
        r += dst_angle - src_angle

        src_coords = AffineTransform(r, (x,y))

        if refine:
            src_mid = self.get_tab_midpoint(src, src_tab_no)
            for _ in range(refine):
                mse, src_coords, sfp, dfp = self.refine_alignment(src, src_coords, src_mid)
        else:
            mse, sfp, dfp = self.measure_fit_fast(src, src_tab_no, src_coords)

        return (mse, src_coords, sfp, dfp)

    def refine_alignment(self, src, src_coords, sfp):

        # starting at src.points[sfp] expand in both directions around the
        # perimeter of the src to find the continuous section that is
        # "close" to dst
        #
        # for each close point find the corresponding (point,normal) in dst
        #
        # compute a least squares optimization of fit

        close_cutoff = 15
        medium_cutoff = 50
    
        n = len(src.points)
        s = 50 # take every s'th point, chosen arbitrarily
        num_samples_each_side = n // (s * 5)
        src_indices = np.arange(sfp - num_samples_each_side * s, sfp + num_samples_each_side * s + 1, s) % n
        src_points = src_coords.get_transform().apply_v2(src.points[src_indices])

        if not self.kdtree:
            self.kdtree = scipy.spatial.KDTree(self.dst.points)
        distance, dst_indices = self.kdtree.query(src_points)

        dst_normals = self.compute_normals(self.dst.points, dst_indices)
        src_normals = src_coords.get_transform().apply_n2(self.compute_normals(src.points, src_indices))
        dot_product = np.sum(dst_normals * src_normals, axis=1)

        is_close = (distance < close_cutoff) | ((distance < medium_cutoff) & (dot_product < -0.9))
        close_points = np.nonzero(is_close)
        if len(close_points[0]) < 2:
            return (None, src_coords, (None, None), (None, None))

        # print(f"{len(src_indices)=} {len(close_points[0])=} {close_points[0]=}")

        close_src_indices = src_indices[close_points]
        close_dst_indices = dst_indices[close_points]

        close_src_points = src.points[close_src_indices]
        close_dst_points = self.dst.points[close_dst_indices]
        close_dst_normals = self.compute_normals(self.dst.points, close_dst_indices)
        
        icp = puzzler.icp.IteratedClosestPoint()
        
        dst_body = icp.make_rigid_body(0., np.array((0., 0.)), fixed=True)
        src_body = icp.make_rigid_body(src_coords.angle, src_coords.dxdy, fixed=False)

        icp.add_body_correspondence(src_body, close_src_points, dst_body, close_dst_points, close_dst_normals)
    
        icp.solve()
    
        self.src_vertexes = close_src_points
        self.dst_vertexes = close_dst_points
        self.dst_normals = close_dst_normals

        src_coords = AffineTransform(src_body.angle, src_body.center)        

        d = self.kdtree.query(src_coords.get_transform().apply_v2(close_src_points))[0]
        mse = np.sum(d ** 2) / len(d)

        d0 = self.kdtree.query(src_coords.get_transform().apply_v2(src.points[src_indices]))[0]
        d1 = self.distance_image.query(src_coords.get_transform().apply_v2(src.points[src_indices]))
        # distances less than zero correspond to points inside the
        # piece, and should always be considered part of the error
        d2 = d1[np.nonzero(is_close | (d1 < 0))]
        mse2 = np.sum(d2 ** 2) / len(d2)

        if False:
            with np.printoptions(precision=3, suppress=True):
                print(f"kdtree: {d=} {mse=}")
                print(f"distance_image: {d0=} {d1=} {d2=} {mse2=}")
                print(np.vstack((d0,d1,is_close)).T)

        # dst is reversed to make ordering respect CW wrapping of
        # points around perimeter
        src_fit_points = (close_src_indices[0], close_src_indices[-1])
        dst_fit_points = (close_dst_indices[-1], close_dst_indices[0])

        return (mse2, src_coords, src_fit_points, dst_fit_points)

    @staticmethod
    def get_tab_midpoint(piece, tab_no):
        tab = piece.tabs[tab_no]
        a, b = tab.fit_indexes
        if a > b:
            n = len(piece.points)
            mid = ((a + b + n) // 2) % n
        else:
            mid = (a + b) // 2

        return mid

    @staticmethod
    def compute_normals(points, indices):

        baseline = 10 # ?
        n = len(points)

        p0 = points[(indices - baseline) % n]
        p1 = points[(indices + baseline) % n]
        v = p1 - p0
        l = np.linalg.norm(v, axis=1)
        # l can be zero when the perimeter is degenerate and the same
        # (x,y) appears multiple times in it, e.g. see tab 3 of piece
        # N34 from the 1000-piece puzzle
        l = np.where(l > 0., l, 1.)
        v = v / l[:,np.newaxis]

        normals = np.array((-v[:,1], v[:,0])).T
        
        return normals
        
    def measure_fit_fast(self, src, src_tab_no, src_coords):
        
        sl, sr = src.tabs[src_tab_no].tangent_indexes

        src_points = src_coords.get_transform().apply_v2(ring_slice(src.points, sl, sr))

        src_fit_points = (sl, sr)
        
        if False:
            d, i = self.kdtree.query(src_points)
            dst_fit_points = (i[-1], i[0])
        else:
            d = self.distance_image.query(src_points)
            dst_fit_points = (None, None)

        if len(d):
            mse = np.sum(d ** 2) / len(d)
        else:
            mse = None

        return (mse, src_fit_points, dst_fit_points)

    def measure_fit(self, src, src_tab_no, src_coords):
        
        thresh = 5

        src_points = src_coords.get_transform().apply_v2(src.points)
        d, i = self.kdtree.query(src_points, distance_upper_bound=thresh+1)

        l, r = src.tabs[src_tab_no].tangent_indexes

        n = len(src.points)

        for j in range(l+n//2, l+n, 1):
            if d[j%n] < thresh:
                break
        l = j % n

        for j in range(r+n//2, r, -1):
            if d[j%n] < thresh:
                break
        r = j % n

        d, _ = self.kdtree.query(ring_slice(src_points, l, r))

        mse = np.sum(d ** 2) / len(d)

        src_fit_points = (l, r)
        dst_fit_points = (i[r], i[l])

        return (mse, src_fit_points, dst_fit_points)

    @staticmethod
    def get_tab_points(piece, tab_no):

        tab = piece.tabs[tab_no]
        ti = list(tab.tangent_indexes)
        l, r = piece.points[ti if tab.indent else ti[::-1]]
        c = tab.ellipse.center
        return np.array([l, c, r])

class EdgeAligner:

    def __init__(self, dst):
        self.dst = dst
        self.kdtree = scipy.spatial.KDTree(dst.points)

    def compute_alignment(self, dst_desc, src, src_desc):
        
        dst_edge = self.dst.edges[dst_desc[0]]
        dst_tab = self.dst.tabs[dst_desc[1]]

        dst_edge_vec = dst_edge.line.pts[1] - dst_edge.line.pts[0]
        dst_edge_angle = np.arctan2(dst_edge_vec[1], dst_edge_vec[0])

        src_edge = src.edges[src_desc[0]]
        src_tab = src.tabs[src_desc[1]]

        src_edge_vec = src_edge.line.pts[1] - src_edge.line.pts[0]
        src_edge_angle = np.arctan2(src_edge_vec[1], src_edge_vec[0])

        src_coords = AffineTransform()
        src_coords.angle = dst_edge_angle - src_edge_angle

        dst_line = dst_edge.line.pts
        src_point = src_coords.get_transform().apply_v2(src_edge.line.pts[0])
        
        src_coords.dxdy = puzzler.math.vector_to_line(src_point, dst_line)

        dst_center = dst_tab.ellipse.center
        src_center = src_coords.get_transform().apply_v2(src_tab.ellipse.center)

        dst_edge_vec = puzzler.math.unit_vector(dst_edge_vec)
        d = np.dot(dst_edge_vec, (dst_center - src_center))
        src_coords.dxdy = src_coords.dxdy + dst_edge_vec * d

        src_fit_pts = (src_tab.tangent_indexes[0], src_edge.fit_indexes[0])
        dst_fit_pts = (dst_edge.fit_indexes[1], dst_tab.tangent_indexes[1])

        # with np.printoptions(precision=3):
        #     print(f"src_coords: angle={src_coords.angle:.3f} xy={src_coords.dxdy}")
        #     print(f"  matrix={src_coords.get_transform().matrix}")

        points = ring_slice(src.points, *src_fit_pts)

        d, _ = self.kdtree.query(src_coords.get_transform().apply_v2(points))

        mse = np.sum(d ** 2) / len(d)

        # print(f"  MSE={mse:.1f}")

        return (mse, src_coords, src_fit_pts, dst_fit_pts)
        
    def get_correspondence(self, src, src_coords, src_fit_indexes):

        a, b = src_fit_indexes

        n = len(src.points)
        if a < b:
            src_indexes = list(range(a,b))
        else:
            src_indexes = list(range(a,n)) + list(range(0,b))

        n = len(src_indexes)
        src_indexes = [src_indexes[i] for i in range(n // 10, n, n // 5)]

        src_points = src.points[src_indexes]

        _, dst_indexes = self.kdtree.query(src_coords.get_transform().apply_v2(src_points))

        return (src_indexes, dst_indexes)

class MultiAligner:

    def __init__(self, targets, pieces, geometry, cache):
        self.targets = targets

        dst_points = []
        for dst_label, dst_tab_no in self.targets:
            dst_piece = pieces[dst_label]
            dst_coords = geometry.coords[dst_label]
            pt = dst_piece.tabs[dst_tab_no].ellipse.center
            dst_points.append(dst_coords.get_transform().apply_v2(pt))

        self.dst_points = np.array(dst_points)

        self.multi_target_error = MultiTargetError(pieces, geometry, cache)

    def compute_alignment(self, src_piece, source_tabs):

        src_points = np.array([src_piece.tabs[s].ellipse.center for s in source_tabs])

        dst_vec = self.dst_points[1] - self.dst_points[0]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[1] - src_points[0]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = AffineTransform(dst_angle-src_angle, (0,0)).get_transform().apply_v2(src_points)

        r, x, y = compute_rigid_transform(src_points_rotated, self.dst_points)

        # with np.printoptions(precision=1):
        #     print(f"{dst_angle=:.3f} {src_angle=:.3f}")
        #     print(f"{src_points=}")
        #     print(f"{src_points_rotated=}")
        #     print(f"{r=:.3f} {x=:.1f} {y=:.1f}")

        r += dst_angle - src_angle

        return AffineTransform(r, (x,y))

    def measure_fit(self, src_piece, source_tabs, src_coords):

        assert 2 == len(source_tabs)
        n_tabs = len(src_piece.tabs)

        tab0, tab1 = source_tabs
        if (tab0 + 1) % n_tabs == tab1:
            l = src_piece.tabs[tab0].tangent_indexes[0]
            r = src_piece.tabs[tab1].tangent_indexes[1]
        elif (tab1 + 1) % n_tabs == tab0:
            l = src_piece.tabs[tab1].tangent_indexes[0]
            r = src_piece.tabs[tab0].tangent_indexes[1]
        else:
            # print(f"measure_fit: {src_piece.label} {n_tabs=} {tab0=} {tab1=}: tabs must be adjacent!")
            return 1e6

        return self.multi_target_error(src_piece, src_coords, l, r)
    
class DistanceQueryCache:

    def __init__(self):
        self.serial_no = 1
        self.cache = dict()
        self.stats = {'n_read':0, 'n_write':0, 'read_miss':0, 'read_hit':0, 'n_purge':0, 'n_bytes':0}

    def query(self, dst_piece, dst_coords, src_piece, src_coords):

        key = self.make_key(dst_piece, dst_coords, src_piece, src_coords)
        if (retval := self.read(key)) is not None:
            return retval

        transform = puzzler.render.Transform()
        transform.rotate(-dst_coords.angle).translate(-dst_coords.dxdy)
        transform.translate(src_coords.dxdy).rotate(src_coords.angle)

        di = DistanceImage.Factory(dst_piece)
        retval = di.query(transform.apply_v2(src_piece.points))
        retval.setflags(write=False)

        self.write(key, retval)

        return retval
        
    def make_key(self, dst_piece, dst_coords, src_piece, src_coords):
        def coords_key(c):
            return (c.angle, c.dxdy[0], c.dxdy[1])
        return (dst_piece.label, coords_key(dst_coords),
                src_piece.label, coords_key(src_coords))

    def read(self, key):
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
    
class MultiTargetError:

    def __init__(self, pieces, geometry, cache):
        self.pieces = pieces
        self.geometry = geometry
        self.overlaps = puzzler.solver.OverlappingPieces(pieces, geometry.coords)
        self.max_dist = 256
        self.distance_query_cache = cache

    def __call__(self, src_piece, src_coords, l, r, verbose=False):

        src_center = src_coords.dxdy
        src_radius = src_piece.radius
        
        dst_labels = self.overlaps(src_center, src_radius + self.max_dist).tolist()
        ret_dist = np.full(len(src_piece.points), self.max_dist)

        if l < r:
            inner = [(l,r)]
            outer = [(0,l), (r,len(ret_dist))]
        else:
            inner = [(l,len(ret_dist)), (0,r)]
            outer = [(r,l)]

        for dst_label in dst_labels:

            dst_piece = self.pieces[dst_label]
            dst_coords = self.geometry.coords[dst_label]

            dst_dist = self.distance_query_cache.query(
                dst_piece, dst_coords, src_piece, src_coords)

            for a, b in inner:
                ret_dist[a:b] = np.minimum(np.abs(dst_dist[a:b]), ret_dist[a:b])

            for a, b in outer:
                ret_dist[a:b] = np.minimum(dst_dist[a:b], ret_dist[a:b])

        sse = 0
        num_points = 0

        if verbose:
            print(f"  <MTE2> {l=} {r=} {inner=} {outer=}")

        for a, b in inner:
            sse += np.sum(np.square(ret_dist[a:b]))
            num_points += b-a

        inner_sse = sse

        for a, b in outer:
            ii = np.nonzero(ret_dist[a:b] < 0)[0]
            if len(ii):
                sse += np.sum(np.square(ret_dist[ii + a]))
                num_points += len(ii)

        if verbose:
            print(f"  <MTE2> {inner_sse=:.1f} {sse=:.1f} {num_points=}")

        return sse / num_points
