import collections
import cv2 as cv
import json
import numpy as np

from contextlib import contextmanager

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

    def __init__(self, lines, props):
        super().__init__(props)
        self.lines = lines

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
    
class SceneGraphBuilder:

    def __init__(self):
        self.stack = [Sequence()]

    def commit(self, camera, viewport):
        node = self.stack.pop()
        return SceneGraph(camera, viewport, node)

    def save(self):
        self.stack.append(Sequence())

    def restore(self):
        node = self.stack.pop()
        self.add_node(node)

    def add_transform(self, m):
        self.add_node(Transform(m))

    def add_translate(self, xy):
        self.add_node(Translate(xy))

    def add_boundingbox(self, bbox, node):
        self.add_node(BoundingBox(bbox, node))

    def add_levelofdetail(self, scales, nodes):
        self.add_node(LevelOfDetail(scales, nodes))

    def add_rotate(self, rad):
        self.add_node(Rotate(rad))

    def add_points(self, points, radius, **kw):
        self.add_node(Points(points, radius, kw))

    def add_lines(self, points, **kw):
        self.add_node(Lines(points, kw))

    def add_circles(self, points, radius, **kw):
        self.add_node(points, radius, kw)

    def add_ellipse(self, center, semi_major, semi_minor, phi, **kw):
        self.add_node(Ellipse(center, semi_major, semi_minor, phi, kw))

    def make_polygon(self, points, **kw):
        return Polygon(points, kw)

    def add_polygon(self, points, **kw):
        self.add_node(Polygon(points, kw))

    def add_text(self, xy, text, **kw):
        self.add_node(Text(xy, text, kw))

    def add_node(self, node):
        self.stack[-1].nodes.append(node)

@contextmanager
def insert_boundingbox(builder, bbox):
    builder.save()
    try:
        yield builder
    finally:
        node = builder.stack.pop()
        builder.add_boundingbox(bbox, node)

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
        self.renderer.rotate(t.rad)

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
        self.renderer.draw_lines(l.lines, **l.props)

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

    def is_bbox_visible(self, bbox):

        ll, ur = bbox
        x0, y0 = ll
        x1, y1 = ur
        points = np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])

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
        self._append({'class':'rotate', 'rad':t.rad})

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
        self._append({'class':'lines', 'lines': l.lines, 'props': l.props})

    def visit_circles(self, c):
        self._append({'class':'circles', 'centers': c.points, 'radius':c.radius, 'props': c.props})

    def visit_ellipse(self, e):
        self._append({'class':'ellipse', 'center': e.center, 'semi_major':e.semi_major, 'semi_minor':e.semi_minor, 'phi':e.phi, 'props':e.props})

    def visit_polygon(self, p):
        self._append({'class':'polygon', 'points':p.points, 'props':p.props})

    def visit_text(self, t):
        self._append({'class':'text', 'xy':t.xy, 'text':t.text, 'props':t.props})

    def _append(self, o):
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

class Predicate:

    def contains(self, pt):
        raise NotImplementedError

class PolygonPredicate(Predicate):

    def __init__(self, bbox, polygon, tags=None):
        self.bbox = bbox
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

    def __init__(self, viewport):
        self.viewport = viewport
        self.cell_w = 64
        self.cell_h = 64
        self.cells = collections.defaultdict(list)
        self.objects = []

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

    def add_object(self, pred):
            
        oid = len(self.objects)
        self.objects.append(pred)

        for cell in self.cells_for_rect(pred.bbox[0], pred.bbox[1]):
            self.cells[cell].append(oid)

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

    def __init__(self, camera, viewport):
        self.hittester = HitTester(viewport)
        self.matrix = np.identity(3)
        self.inverse = np.identity(3)

        self.matrix = camera
        self.inverse = np.linalg.inv(self.matrix)

    def __call__(self, node):
        node.accept(self)
        return self.hittester

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
        pass

    def visit_polygon(self, p):
        pred = PolygonPredicate(compute_bounding_box(p.points), p.points, p.props.get('tags'))
        pred = TransformPredicate(self.matrix, self.inverse, pred)
        self.hittester.add_object(pred)

    def visit_text(self, n):
        pass

