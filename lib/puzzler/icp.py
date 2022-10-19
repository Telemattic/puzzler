import math
import numpy as np

class IteratedClosestPoint:

    class RigidBody:

        def __init__(self, i, angle, center, fixed):
            self.index  = i
            self.angle  = angle
            self.center = center
            self.fixed  = fixed

    def __init__(self):
        self.n_cols = 0
        self.bodies = []
        self.data = []
        self.verbose = True

    def make_rigid_body(self, angle, center=(0.,0.), fixed=False):

        i = None
        if not fixed:
            i = self.n_cols
            self.n_cols += 3
            center = np.array((0, 0))
            
        body = IteratedClosestPoint.RigidBody(i, angle, center, fixed)
        self.bodies.append(body)

        return body
        
    def add_correspondence(self, src, src_vertex, dst, dst_vertex, dst_normal):

        assert isinstance(src, IteratedClosestPoint.RigidBody)
        assert isinstance(dst, IteratedClosestPoint.RigidBody)
        assert not (src is dst)
        assert not src.fixed

        self.data.append((src, src_vertex.copy(), dst, dst_vertex.copy(), dst_normal.copy()))

    def data_to_A_b(self, src, src_vertex, dst, dst_vertex, dst_normal):
        
        def rotation_matrix(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array(((c, -s), (s, c)))

        src_matrix = rotation_matrix(src.angle)
        src_vertex = src_vertex @ src_matrix.T

        dst_matrix = rotation_matrix(dst.angle)
        dst_vertex = dst_vertex @ dst_matrix.T
        if dst.fixed:
            dst_vertex += dst.center
        dst_normal = dst_normal @ dst_matrix.T
        
        a_ij = np.cross(src_vertex, dst_normal)
        a_ji = np.cross(dst_vertex, dst_normal)
        # row-wise dot product
        b_ij = np.sum((src_vertex - dst_vertex) * dst_normal, axis=1)
        n_ij = dst_normal

        return (src.index, dst.index, a_ij, a_ji, n_ij, b_ij)

    def solve(self):

        n_rows = sum(len(i[1]) for i in self.data)
        n_cols = self.n_cols

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows))

        r = 0
        for v in self.data:
            
            i, j, a_ij, a_ji, n_ij, b_ij = self.data_to_A_b(*v)

            k = len(a_ij)
            assert k == len(n_ij) == len(b_ij)
            
            A[r:r+k,i] = a_ij
            A[r:r+k,i+1:i+3] = n_ij

            if j is not None:
                A[r:r+k,j] = -a_ji
                A[r:r+k,j+1:j+3] = -n_ij

            b[r:r+k] = b_ij
            r += k

        assert r == n_rows

        if self.verbose:
            with np.printoptions(precision=2, linewidth=120):
                print(f"{A.shape=} {A=}")
                print(f"{b.shape=} {b=}")

        # minimize (Ax-b)**2
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        if self.verbose:
            with np.printoptions(precision=3, linewidth=120):
                print(f"{x.shape=} {x=}")

        c = 0
        for body in self.bodies:

            if body.fixed:
                continue

            body.angle  = body.angle - x[c]
            body.center = - np.array((x[c+1], x[c+2]))
            c += 3

        assert c == n_cols
