import puzzler.scenegraph as sg
import cv2 as cv
import numpy as np
from contextlib import contextmanager

def simplify_line(points, epsilon):
    approx = cv.approxPolyDP(points, epsilon, closed=False)
    return np.squeeze(approx)

def simplify_polygon(points, epsilon):
    approx = cv.approxPolyDP(points, epsilon, closed=True)
    return np.squeeze(approx)
# poly = np.squeeze(approx)
# return np.concatenate((poly, poly[:1,:]))

class SceneGraphBuilder:

    def __init__(self):
        self.stack = [sg.Sequence()]

    def commit(self, camera, viewport):
        node = self.stack.pop()
        return sg.SceneGraph(camera, viewport, node)

    def sequence_begin(self):
        self.stack.append(sg.Sequence())

    def sequence_end(self):
        node = self.stack.pop()
        self.add_node(node)

    def boundingbox_begin(self):
        self.stack.append(sg.Sequence())

    def boundingbox_end(self, bbox):
        node = self.stack.pop()
        self.add_boundingbox(bbox, node)

    def add_transform(self, m):
        self.add_node(sg.Transform(m))

    def add_translate(self, xy):
        self.add_node(sg.Translate(xy))

    def add_boundingbox(self, bbox, node):
        self.add_node(sg.BoundingBox(bbox, node))

    def add_levelofdetail(self, scales, nodes):
        self.add_node(sg.LevelOfDetail(scales, nodes))

    def add_rotate(self, rad):
        self.add_node(sg.Rotate(rad))

    def add_points(self, points, radius, **kw):
        # self.add_node(Points(points, radius, kw))
        kw['radius'] = radius
        self.add_node(sg.Points(points, kw))
        
    def add_lines(self, points, **kw):
        self.add_node(sg.Lines(sg.make_array_CV_32(points), kw))

    def add_circles(self, points, radius, **kw):
        self.add_node(sg.Circles(points, radius, kw))

    def add_ellipse(self, center, semi_major, semi_minor, phi, **kw):
        self.add_node(sg.Ellipse(center, semi_major, semi_minor, phi, kw))

    def make_polygon(self, points, **kw):
        return sg.Polygon(points, kw)

    def add_polygon(self, points, **kw):
        self.add_node(sg.Polygon(sg.make_array_CV_32(points), kw))

    def add_text(self, xy, text, **kw):
        self.add_node(sg.Text(xy, text, kw))

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

class LevelOfDetailFactory(sg.SceneGraphCloner):

    def __init__(self):
        super().__init__()
        self.scales = (0.2, 0.6)
        self.epsilons = [0, 2, 5]

    def visit_levelofdetail(self, l):
        # don't visit the children of an LOD node, as we've already
        # got an LOD node
        self.append(l)

    def visit_lines(self, l):
        if len(l.points) < 10:
            self.append(l)
            return
        
        nodes = []
        for eps in self.epsilons:
            points = simplify_line(l.points, eps)
            nodes.append(sg.Lines(points, l.props))

        self.append(sg.LevelOfDetail(self.scales, nodes))

    def visit_polygon(self, p):
        
        nodes = []
        for eps in self.epsilons:
            points = simplify_polygon(p.points, eps)
            nodes.append(sg.Polygon(points, p.props))

        self.append(sg.LevelOfDetail(self.scales, nodes))

class PieceSceneGraphFactory:

    defaults = {
        'tabs.render':True,
        'tabs.ellipse.fill':'cyan',
        'tabs.ellipse.outline':'',
        'tabs.ellipse.dashes':(6, 6),
        'tabs.ellipse.width':2,
        'tabs.label.font':('Courier New', 12),
        'tabs.label.fill':'darkblue',
        'edges.render':True,
        'edges.lines.fill':'pink',
        'edges.lines.width':4,
        'points.outline':'black',
        'points.fill':'',
        'points.width':1,
        'label.render':True,
        'label.font':('Courier New', 18),
        'label.fill':'black'
    }
    
    def __init__(self, pieces, opt = dict()):
        self.pieces = pieces
        self.opt = PieceSceneGraphFactory.defaults | opt
        self.levelofdetail = LevelOfDetailFactory()
        self._cache = dict()
        self.nodes = []

    def __call__(self, label, props = dict()):
        key = tuple(sorted(({'label':label} | props).items()))
        node = self._cache.get(key)
        if node is None:
            save_opt = self.opt
            self.opt = self.opt | props
            self._cache[key] = node = self.do_piece(self.pieces[label])
            self.opt = save_opt
        return node

    def do_piece(self, p):

        self.nodes = []

        if self.opt['tabs.render'] and p.tabs:
            self.do_tabs(p)

        if self.opt['edges.render'] and p.edges:
            self.do_edges(p)

        self.do_outline(p)

        if self.opt['label.render']:
            self.do_label(p)

        bbox = sg.compute_bounding_box(p.points)

        return sg.BoundingBox(bbox, sg.Sequence(self.nodes))

    def do_tabs(self, p):

        tags = self.opt.get('tags', (p.label,))

        e_fill = self.opt['tabs.ellipse.fill']
        e_outline = self.opt['tabs.ellipse.outline']
        e_dashes = self.opt['tabs.ellipse.dashes']
        e_width = self.opt['tabs.ellipse.width']

        l_font = self.opt['tabs.label.font']
        l_fill = self.opt['tabs.label.fill']

        for i, tab in enumerate(p.tabs):
            e = tab.ellipse
            self.add_node(sg.Ellipse(e.center, e.semi_major, e.semi_minor, e.phi,
                                     {'fill':e_fill, 'outline':e_outline, 'dashes':e_dashes, 'width':e_width, 'tags':tags}))
            self.add_node(sg.Text(e.center, str(i), {'font':l_font, 'fill':l_fill}))

    def do_edges(self, p):

        props = {'width': self.opt['edges.lines.width'],
                 'fill': self.opt['edges.lines.fill']}

        for edge in p.edges:
            points = sg.make_array_CV_32(edge.line.pts)
            self.add_node(sg.Lines(points, props))

    def do_outline(self, p):

        tags = self.opt.get('tags', (p.label,))
        props = {'outline': self.opt['points.outline'],
                 'fill': self.opt['points.fill'],
                 'width': self.opt['points.width'],
                 'tags':tags}
        polygon = sg.Polygon(p.points, props)
        lod = self.levelofdetail.visit_node(polygon)
        self.add_node(lod)

    def do_label(self, p):

        props = {'font': self.opt['label.font'],
                 'fill': self.opt['label.fill']}
        self.add_node(sg.Text(np.zeros(2), p.label, props))

    def add_node(self, n):
        self.nodes.append(n)
