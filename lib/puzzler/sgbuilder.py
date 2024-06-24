import puzzler.scenegraph as sg
import cv2 as cv
import numpy as np
from contextlib import contextmanager

def simplify_polygon(points, epsilon):
    approx  = cv.approxPolyDP(points, epsilon, True)
    poly = np.squeeze(approx)
    return np.concatenate((poly, poly[:1,:]))

class SceneGraphBuilder:

    def __init__(self):
        self.stack = [Sequence()]

    def commit(self, camera, viewport):
        node = self.stack.pop()
        return SceneGraph(camera, viewport, node)

    def sequence_begin(self):
        self.stack.append(Sequence())

    def sequence_end(self):
        node = self.stack.pop()
        self.add_node(node)

    def boundingbox_begin(self):
        self.stack.append(Sequence())

    def boundingbox_end(self, bbox):
        node = self.stack.pop()
        self.add_boundingbox(bbox, node)

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
def insert_sequence(builder):
    builder.sequence_begin()
    try:
        yield builder
    finally:
        builder.sequence_end()
    
@contextmanager
def insert_boundingbox(builder, bbox):
    builder.boundingbox_begin()
    try:
        yield builder
    finally:
        builder.boundingbox_end(bbox)

class PieceSceneGraphFactory:

    defaults = {
        'tabs.render':True,
        'tabs.ellipse.fill':'cyan',
        'tabs.ellipse.outline':'',
        'tabs.label.font':('Courier New', 12),
        'tabs.label.fill':'darkblue',
        'edges.render':True,
        'edges.lines.fill':'pink',
        'edges.lines.width':4,
        'points.outline':'black',
        'points.fill':'',
        'points.width':1,
        'label.font':('Courier New', 18),
        'label.fill':'black'
    }
    
    def __init__(self, pieces, **kw):
        self.pieces = pieces
        self.opt = PieceSceneGraphFactory.defaults | kw
        self._cache = dict()
        self.nodes = []

    def __call__(self, label):
        node = self._cache.get(label)
        if node is None:
            self._cache[label] = node = self.do_piece(self.pieces[label])
        return node

    def do_piece(self, p):

        self.nodes = []

        if self.opt['tabs.render'] and p.tabs:
            self.do_tabs(p)

        if self.opt['edges.render'] and p.edges:
            self.do_edges(p)

        self.do_outline(p)

        self.do_label(p)

        bbox = sg.compute_bounding_box(p.points)

        return sg.BoundingBox(bbox, sg.Sequence(self.nodes))

    def do_tabs(self, p):

        e_fill = self.opt['tabs.ellipse.fill']
        e_outline = self.opt['tabs.ellipse.outline']

        l_font = self.opt['tabs.label.font']
        l_fill = self.opt['tabs.label.fill']

        for i, tab in enumerate(p.tabs):
            e = tab.ellipse
            self.add_node(sg.Ellipse(e.center, e.semi_major, e.semi_minor, e.phi,
                                     {'fill':e_fill, 'outline':e_outline, 'tags':(p.label,)}))
            self.add_node(sg.Text(e.center, str(i), {'font':l_font, 'fill':l_fill}))

    def do_edges(self, p):

        l_width = self.opt['edges.lines.width']
        l_fill = self.opt['edges.lines.fill']

        for edge in p.edges:
            self.add_node(sg.Lines(edge.line.pts, {'width':l_width, 'fill':l_fill}))

    def do_outline(self, p):

        props = {'outline': self.opt['points.outline'],
                 'fill': self.opt['points.fill'],
                 'width': self.opt['points.width'],
                 'tags':(p.label,)}
        scales = [0.2, 0.06]
        nodes = [sg.Polygon(simplify_polygon(p.points, 0), props),
                 sg.Polygon(simplify_polygon(p.points, 2), props),
                 sg.Polygon(simplify_polygon(p.points, 5), props)]
        self.add_node(sg.LevelOfDetail(scales, nodes))

    def do_label(self, p):

        props = {'font': self.opt['label.font'],
                 'fill': self.opt['label.fill']}
        self.add_node(sg.Text(np.zeros(2), p.label, props))

    def add_node(self, n):
        self.nodes.append(n)
