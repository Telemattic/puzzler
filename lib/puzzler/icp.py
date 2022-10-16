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

        def rotation_matrix(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array(((c, -s), (s, c)))

        src_matrix = rotation_matrix(src.angle)
        src_vertex = src_vertex @ src_matrix.T

        dst_matrix = rotation_matrix(dst.angle)
        dst_vertex = dst_vertex @ dst_matrix.T
        if dst.fixed:
            dst_vertex += dst.center
        dst_normal = dst_vertex @ dst_matrix.T
            
        a_ij = np.cross(src_vertex, dst_normal)
        # row-wise dot product
        b_ij = np.sum((src_vertex - dst_vertex) * dst_normal, axis=1)
        n_ij = dst_normal.copy()

        self.data.append((src.index, dst.index, a_ij, n_ij, b_ij))

    def solve(self):

        n_rows = sum(len(i[2]) for i in self.data)
        n_cols = self.n_cols

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows))

        r = 0
        for v in self.data:
            
            i, j, a_ij, n, b_ij = v

            k = len(a_ij)
            assert k == len(n) == len(b_ij)
            
            A[r:r+k,i] = a_ij
            A[r:r+k,i+1:i+3] = n

            if j is not None:
                A[r:r+k,j] = -a_ij
                A[r:r+k,j+1:j+3] = -n

            b[r:r+k] = b_ij

        # minimize (Ax-b)**2
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        for body in self.bodies:

            if body.fixed:
                continue

            body.angle += x[i]
            body.center = np.array((x[i+1], x[i+2]))
