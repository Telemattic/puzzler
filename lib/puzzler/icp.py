import math
import numpy as np
from dataclasses import dataclass

class IteratedClosestPoint:

    @dataclass
    class RigidBody:
        index: int
        angle: float
        center: np.array
        fixed: bool

    @dataclass
    class Axis:
        index: int
        normal: np.array
        value: float
        fixed: bool

    def __init__(self, verbose=False):
        self.n_cols = 0
        self.bodies = []
        self.axes = []
        self.data = []
        self.verbose = verbose

    def make_rigid_body(self, angle, center=(0.,0.), fixed=False):

        i = None
        if not fixed:
            i = self.n_cols
            self.n_cols += 3
            center = np.array((0, 0))
            
        body = IteratedClosestPoint.RigidBody(i, angle, center, fixed)
        self.bodies.append(body)

        return body

    def make_axis(self, normal, value=0., fixed=False):

        i = None
        if not fixed:
            i = self.n_cols
            self.n_cols += 1
            value = 0.

        axis = IteratedClosestPoint.Axis(i, normal, value, fixed)
        self.axes.append(axis)

        return axis

    def add_axis_correspondence(self, src, src_vertex, dst):

        assert isinstance(src, IteratedClosestPoint.RigidBody)
        assert isinstance(dst, IteratedClosestPoint.Axis)
        assert not src.fixed

        self.data.append(('axis', src, src_vertex.copy(), dst))
        
    def add_body_correspondence(self, src, src_vertex, dst, dst_vertex, dst_normal):

        assert isinstance(src, IteratedClosestPoint.RigidBody)
        assert isinstance(dst, IteratedClosestPoint.RigidBody)
        assert not (src is dst)
        assert not src.fixed

        self.data.append(('body', src, src_vertex.copy(), dst, dst_vertex.copy(), dst_normal.copy()))

    def body_data_to_A_b(self, src, src_vertex, dst, dst_vertex, dst_normal):
        
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

        A_terms = [(src.index, a_ij), (src.index+1, n_ij[:,0]), (src.index+2, n_ij[:,1])]
        if not dst.fixed:
            A_terms += [(dst.index, -a_ji), (dst.index+1, -n_ij[:,0]), (dst.index+2, -n_ij[:,1])]

        b_term = b_ij

        return A_terms, b_term

    def axis_data_to_A_b(self, src, src_vertex, dst):

        def rotation_matrix(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array(((c, -s), (s, c)))

        src_matrix = rotation_matrix(src.angle)
        src_vertex = src_vertex @ src_matrix.T

        k = len(src_vertex)

        A_terms = [(src.index, np.cross(src_vertex, dst.normal)),
                   (src.index+1, np.full(k, dst.normal[0])),
                   (src.index+2, np.full(k, dst.normal[1]))]
        b_term = np.sum(src_vertex * dst.normal, axis=1)
        
        if dst.fixed:
            b_term -= dst.value
        else:
            A_terms += [(dst.index, np.full(k, -1.))]

        return A_terms, b_term

    def solve(self):

        n_rows = sum(len(i[2]) for i in self.data)
        n_cols = self.n_cols

        A = np.zeros((n_rows, n_cols))
        b = np.zeros((n_rows))

        r = 0
        for v in self.data:

            if v[0] == 'body':

                A_terms, b_term = self.body_data_to_A_b(*v[1:])

            elif v[0] == 'axis':

                A_terms, b_term = self.axis_data_to_A_b(*v[1:])

            for i, a in A_terms:
                k = len(a)
                A[r:r+k,i] = a

            k = len(b_term)
            b[r:r+k] = b_term

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

        for body in self.bodies:

            if body.fixed:
                continue

            body.angle  = body.angle - x[body.index]
            body.center = - np.array((x[body.index+1], x[body.index+2]))

        for axis in self.axes:

            if axis.fixed:
                continue

            axis.value = - x[axis.index]

            if self.verbose:
                print(f"{axis.index=} {axis.value=:.3f}")
