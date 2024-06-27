import collections
import cv2 as cv
import json
import math
import numpy as np

from contextlib import contextmanager

def is_array_CV_32(data):

    return isinstance(data, np.ndarray) and data.dtype in (np.int32, np.float32)

def make_array_CV_32(data):

    # cv::approxPolyDP and cv::pointPolygonTest require data that is
    # CV_32F or CV_32S, make it so
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.dtype in (np.int32, np.float32):
        return data
        
    dtype = np.int32 if data.dtype.kind in 'iu' else np.float32
    return np.array(data, dtype=dtype)

class SceneGraph:

    def __init__(self, camera_matrix, viewport, root_node):
        self.camera_matrix = camera_matrix
        self.viewport = viewport
        self.root_node = root_node

class Node:

    def __init__(self):
        pass

class Sequence(Node):

    def __init__(self, nodes=None):
        if nodes is None:
            nodes = []
        self.nodes = nodes

    def accept(self, v):
        v.visit_sequence(self)

class Transform(Node):

    def __init__(self, matrix):
        self.matrix = matrix

    def accept(self, v):
        v.visit_transform(self)

class Translate(Node):

    def __init__(self, xy):
        self.xy = xy

    def accept(self, v):
        v.visit_translate(self)

class Rotate(Node):

    def __init__(self, rad):
        self.rad = rad

    def accept(self, v):
        v.visit_rotate(self)

class BoundingBox(Node):

    def __init__(self, bbox, node):
        self.bbox = bbox
        self.node = node

    def accept(self, v):
        v.visit_boundingbox(self)

class LevelOfDetail(Node):

    def __init__(self, scales, nodes):
        self.scales = scales
        self.nodes = nodes

    def accept(self, v):
        v.visit_levelofdetail(self)

class Geometry(Node):

    def __init__(self, props):
        self.props = props

class Points(Geometry):

    def __init__(self, points, props):
        super().__init__(props)
        self.points = points

    def accept(self, v):
        v.visit_points(self)

class Lines(Geometry):

    def __init__(self, points, props):
        assert is_array_CV_32(points)
        super().__init__(props)
        self.points = points

    def accept(self, v):
        v.visit_lines(self)

class Circles(Geometry):

    def __init__(self, points, radius, props):
        super().__init__(props)
        self.points = points
        self.radius = radius

    def accept(self, v):
        v.visit_circles(self)

class Ellipse(Geometry):

    def __init__(self, center, semi_major, semi_minor, phi, props):
        super().__init__(props)
        self.center = center
        self.semi_major = semi_major
        self.semi_minor = semi_minor
        self.phi = phi

    def accept(self, v):
        v.visit_ellipse(self)

def compute_bounding_box(points):
    return (np.min(points, 0), np.max(points, 0))

class Polygon(Geometry):

    def __init__(self, points, props):
        assert is_array_CV_32(points)
        super().__init__(props)
        self.points = points

    def accept(self, v):
        v.visit_polygon(self)

class Text(Geometry):
                       
    def __init__(self, xy, text, props):
        super().__init__(props)
        self.xy = xy
        self.text = text

    def accept(self, v):
        v.visit_text(self)

class SceneGraphVisitor:

    def visit_sequence(self, n):
        raise NotImplementedError

    def visit_transform(self, n):
        raise NotImplementedError

    def visit_translate(self, n):
        raise NotImplementedError

    def visit_rotate(self, n):
        raise NotImplementedError

    def visit_boundingbox(self, n):
        raise NotImplementedError

    def visit_levelofdetail(self, n):
        raise NotImplementedError

    def visit_points(self, n):
        raise NotImplementedError
    
    def visit_lines(self, n):
        raise NotImplementedError
    
    def visit_circles(self, n):
        raise NotImplementedError
    
    def visit_ellipse(self, n):
        raise NotImplementedError
    
    def visit_polygon(self, n):
        raise NotImplementedError
    
    def visit_text(self, n):
        raise NotImplementedError

class SceneGraphCloner:

    def __init__(self):
        self._stack = []

    def visit_node(self, node):
        node.accept(self)
        return self._stack.pop()

    def visit_nodes(self, nodes):
        i = len(self._stack)
        for n in nodes:
            n.accept(self)
        retval = self._stack[i:]
        self._stack = self._stack[:i]
        return retval

    def visit_sequence(self, n):
        children = self.visit_nodes(n.nodes)
        self.append(Sequence(children))

    def visit_transform(self, n):
        self.append(n)

    def visit_translate(self, n):
        self.append(n)

    def visit_rotate(self, n):
        self.append(n)

    def visit_boundingbox(self, n):
        child = self.visit_node(n.node)
        self.append(BoundingBox(n.bbox, child))

    def visit_levelofdetail(self, n):
        children = self.visit_nodes(n.nodes)
        self.append(LevelOfDetail(n.scales, children))

    def visit_points(self, n):
        self.append(n)
    
    def visit_lines(self, n):
        self.append(n)
    
    def visit_circles(self, n):
        self.append(n)
    
    def visit_ellipse(self, n):
        self.append(n)
    
    def visit_polygon(self, n):
        self.append(n)
    
    def visit_text(self, n):
        self.append(n)

    def append(self, n):
        self._stack.append(n)
    
class SceneGraphRenderer(SceneGraphVisitor):

    def __init__(self, renderer, viewport, scale=1.):
        self.renderer = renderer
        self.viewport = viewport
        self.scale = scale
        self.fonts = dict()
        self.total_poly_points = 0

    def visit_sequence(self, s):
        self.renderer.save()
        for node in s.nodes:
            node.accept(self)
        self.renderer.restore()

    def visit_transform(self, t):
        self.renderer.transform(t.matrix)

    def visit_translate(self, t):
        self.renderer.translate(t.xy)

    def visit_rotate(self, r):
        self.renderer.rotate(r.rad)

    def visit_boundingbox(self, b):
        if self.is_bbox_visible(b.bbox):
            b.node.accept(self)

    def visit_levelofdetail(self, l):
        i = 0
        while i < len(l.scales) and self.scale < l.scales[i]:
            i += 1
        l.nodes[i].accept(self)

    def visit_points(self, p):
        self.renderer.draw_points(p.points, **p.props)

    def visit_lines(self, l):
        self.renderer.draw_lines(l.points, **l.props)

    def visit_circles(self, c):
        self.renderer.draw_circles(c.points, c.radius, **c.props)

    def visit_ellipse(self, e):
        self.renderer.draw_ellipse(e.center, e.semi_major, e.semi_minor, e.phi, **e.props)

    def visit_polygon(self, p):
        self.renderer.draw_polygon(p.points, **p.props)
        self.total_poly_points += len(p.points)

    def visit_text(self, t):
        if 'font' in t.props:
            props = t.props.copy()
            props['font'] = self.make_font(props['font'])
        else:
            props = t.props
        self.renderer.draw_text(t.xy, t.text, **props)

    def make_font(self, fontspec):
        font = self.fonts.get(fontspec)
        if font is None:
            font = self.renderer.make_font(*fontspec)
            self.fonts[fontspec] = font
        return font

    def bbox_corners(self, bbox):
        ll, ur = bbox
        x0, y0 = ll
        x1, y1 = ur
        return np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])

    def is_bbox_visible(self, bbox):

        points = self.bbox_corners(bbox)
        screen = self.renderer.user_to_device(points)
        x = screen[:,0]
        y = screen[:,1]

        if np.max(x) < 0 or np.min(x) > self.viewport[0]:
            return False

        if np.max(y) < 0 or np.min(y) > self.viewport[1]:
            return False

        return True
    
class SceneGraphFormatter(SceneGraphVisitor):

    def __init__(self):
        self.stack = []

    def __call__(self, sg):
        self.stack.append([])
        sg.root_node.accept(self)
        n = self.stack.pop()
        return {'class':'scenegraph', 'camera_matrix':sg.camera_matrix, 'viewport':sg.viewport, 'root_node':n[0]}

    def visit_sequence(self, s):
        self.stack.append([])
        for n in s.nodes:
            n.accept(self)
        self._append({'class':'sequence', 'nodes':self.stack.pop()})

    def visit_transform(self, t):
        self._append({'class':'transform', 'matrix':t.matrix})

    def visit_translate(self, t):
        self._append({'class':'translate', 'xy':t.xy})

    def visit_rotate(self, r):
        self._append({'class':'rotate', 'rad':r.rad})

    def visit_boundingbox(self, b):
        self.stack.append([])
        b.node.accept(self)
        nodes = self.stack.pop()
        self._append({'class':'boundingbox', 'bbox':b.bbox, 'node':nodes[0]})

    def visit_levelofdetail(self, l):
        self.stack.append([])
        for n in l.nodes:
            n.accept(self)
        nodes = self.stack.pop()
        self._append({'class':'levelofdetail', 'scales':l.scales, 'nodes':nodes})

    def visit_points(self, p):
        self._append({'class':'points', 'points': p.points, 'props': p.props})

    def visit_lines(self, l):
        self._append({'class':'lines', 'points': l.points, 'props': l.props})

    def visit_circles(self, c):
        self._append({'class':'circles', 'centers': c.points, 'radius':c.radius, 'props': c.props})

    def visit_ellipse(self, e):
        self._append({'class':'ellipse', 'center': e.center, 'semi_major':e.semi_major, 'semi_minor':e.semi_minor, 'phi':e.phi, 'props':e.props})

    def visit_polygon(self, p):
        self._append({'class':'polygon', 'points':p.points, 'props':p.props})

    def visit_text(self, t):
        self._append({'class':'text', 'xy':t.xy, 'text':t.text, 'props':t.props})

    def _append(self, o):
        if 'points' in o:
            o = o | {'points':str(o['points'].shape)}
        self.stack[-1].append(o)

class SceneGraphJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def to_json(sg):
    o = SceneGraphFormatter()(sg)
    return SceneGraphJSONEncoder(indent=0).encode(o)

def project_point(matrix, pt):
    pt = np.array((*pt, 1.)) @ matrix.T
    return pt[:2]

def project_bbox(matrix, bbox):
    
    ll, ur = bbox
    x0, y0 = ll
    x1, y1 = ur
    points = np.array([(x0, y0, 1.), (x1, y0, 1.), (x1, y1, 1.), (x0, y1, 1.)])
    points = points @ matrix.T
    x, y = points[:,0], points[:,1]
    x0, y0 = np.min(x), np.min(y)
    x1, y1 = np.max(x), np.max(y)
    return ((x0, y0), (x1, y1))

def bbox_contains(bbox, pt):
    ll, ur = bbox
    return ll[0] <= pt[0] <= ur[0] and ll[1] <= pt[1] <= ur[1]

def bbox_union(bboxes):
    x0, y0 = bboxes[0][0]
    x1, y1 = bboxes[0][1]

    for ll, ur in bboxes[1:]:
        x0 = min(x0, ll[0])
        y0 = min(y0, ll[1])
        x1 = max(x1, ur[0])
        y1 = max(y1, ur[1])

    return ((x0, y0), (x1, y1))

class Predicate:

    def contains(self, pt):
        raise NotImplementedError

class EllipsePredicate(Predicate):

    def __init__(self, center, semi_major, semi_minor, phi, tags=None):
        
        c, s = math.cos(-phi), math.sin(-phi)
        self.rot = np.array(((c, s), (-s, c)))
        w, h = semi_major, semi_minor
        points = center + (np.array([(-w, -h), (w, -h), (w, h), (-w, h)]) @ self.rot)
        x, y = points[:,0], points[:,1]
        x0, y0 = np.min(x), np.min(y)
        x1, y1 = np.max(x), np.max(y)
        self.bbox = ((x0, y0), (x1, y1))
        self.center = center
        self.semi_major = semi_major
        self.semi_minor = semi_minor
        self.phi = phi
        self._tags = tags

    def tags(self):
        return self._tags

    def contains(self, pt):
        return bbox_contains(self.bbox, pt) and self.point_in_ellipse(pt)

    def point_in_ellipse(self, pt):
        x, y = (pt - self.center) @ self.rot # probably need inverse here
        w, h = self.semi_major, self.semi_minor
        w2, h2 = w * w, h * h
        # (x/w)^2 + (y/h)^2 <= 1, multiply by (w^2)*(h^2) to eliminate division
        return (x * x) * h2  + (y * y) * w2 < w2 * h2

class PolygonPredicate(Predicate):

    def __init__(self, polygon, tags=None):
        # required for pointPolygonTest
        assert is_array_CV_32(polygon)
        self.bbox = compute_bounding_box(polygon)
        self.polygon = polygon
        self._tags = tags

    def tags(self):
        return self._tags

    def contains(self, pt):
        return bbox_contains(self.bbox, pt) and 0. <= cv.pointPolygonTest(self.polygon, pt, measureDist=False)

class TransformPredicate(Predicate):

    def __init__(self, matrix, inverse, delegate):
        self.matrix = matrix
        self.inverse = inverse
        self.bbox = project_bbox(self.matrix, delegate.bbox)
        self.delegate = delegate

    def tags(self):
        return self.delegate.tags()

    def contains(self, pt):
        return bbox_contains(self.bbox, pt) and self.delegate.contains(project_point(self.inverse, pt))

class HitTester:

    def __init__(self, cell_size, objects):
        self.cell_w, self.cell_h = cell_size
        self.cells = collections.defaultdict(list)
        self.objects = objects

        for oid, o in enumerate(self.objects):
            for cell in self.cells_for_rect(o.bbox[0], o.bbox[1]):
                self.cells[cell].append(oid)

    def __call__(self, pt):

        cell = self.cell_for_point(pt)
        if cell not in self.cells:
            return []
        
        retval = []
        for oid in self.cells[cell]:
            o = self.objects[oid]
            if o.contains(pt):
                retval.append((oid, o.tags()))
        return retval

    def cell_for_point(self, xy):
        return int(xy[0]) // self.cell_w, int(xy[1]) // self.cell_h
    
    def cells_for_rect(self, pt0, pt1):
        x0, y0 = self.cell_for_point(pt0)
        x1, y1 = self.cell_for_point(pt1)
        for i in range(x0,x1+1):
            for j in range(y0,y1+1):
                yield (i,j)

def translate_matrix(xy):
    x, y = xy
    return np.array(((1, 0, x),
                     (0, 1, y),
                     (0, 0, 1)))

def rotate_matrix(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array(((c, -s, 0),
                     (s,  c, 0),
                     (0,  0, 1)))
            
class BuildHitTester(SceneGraphVisitor):

    def __init__(self):
        self.matrix = np.identity(3)
        self.inverse = np.identity(3)

    def __call__(self, node):
        self.objects = []
        node.accept(self)
        return HitTester(self.compute_cell_size(), self.objects)

    def compute_cell_size(self):
        bbox = bbox_union([o.bbox for o in self.objects])
        w, h = int(bbox[1][0]-bbox[0][0]), int(bbox[1][1]-bbox[0][1])

        # (w/d) * (h/d) ~= n --> d = sqrt(w*h/n)
        w = max(1, w)
        h = max(1, h)
        d = int(math.sqrt(w * h / 1024))

        return (d, d)


    def visit_sequence(self, s):
        
        prev_matrix = self.matrix
        prev_inverse = self.inverse
        
        for n in s.nodes:
            n.accept(self)

        self.inverse = prev_inverse
        self.matrix = prev_matrix

    def visit_transform(self, t):
        self.matrix = self.matrix @ t.matrix
        self.inverse = np.linalg.inv(self.matrix)

    def visit_translate(self, t):
        self.matrix = self.matrix @ translate_matrix(t.xy)
        self.inverse = translate_matrix(-np.array(t.xy)) @ self.inverse

    def visit_rotate(self, t):
        self.matrix = self.matrix @ rotate_matrix(t.rad)
        self.inverse = rotate_matrix(-t.rad) @ self.inverse

    def visit_boundingbox(self, b):
        b.node.accept(self)

    def visit_levelofdetail(self, n):
        # just take the highest LOD to test against?
        n.nodes[0].accept(self)

    def visit_points(self, n):
        pass

    def visit_lines(self, n):
        pass

    def visit_circles(self, n):
        pass

    def visit_ellipse(self, n):
        pred = EllipsePredicate(n.center, n.semi_major, n.semi_minor, n.phi, n.props.get('tags'))
        pred = TransformPredicate(self.matrix, self.inverse, pred)
        self.objects.append(pred)

    def visit_polygon(self, p):
        pred = PolygonPredicate(p.points, p.props.get('tags'))
        pred = TransformPredicate(self.matrix, self.inverse, pred)
        self.objects.append(pred)

    def visit_text(self, n):
        pass

