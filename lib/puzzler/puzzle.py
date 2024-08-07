import puzzler

import numpy as np
import re
from dataclasses import dataclass, field

@dataclass
class Puzzle:

    @dataclass
    class Scan:

        path: str

    @dataclass
    class Piece:

        @dataclass
        class Source:
            id: str
            rect: "tuple[int, int, int, int]"

        def __post_init__(self):
            if self.points is not None and len(self.points):
                self.bbox = (np.min(self.points, axis=0), np.max(self.points, axis=0))
                self.radius = np.max(np.linalg.norm(self.points, axis=1))
            else:
                self.bbox = (np.zeros(2), np.zeros(2))
                self.radius = 0.
            
        label:  str
        source: Source
        points: np.array
        tabs:   "list[puzzler.feature.Tab]"
        edges:  "list[puzzler.feature.Edge]"
        bbox:   "tuple[np.array, np.array]" = field(init=False)
        radius: "float" = field(init=False)

    scans: "dict[str, Scan]"
    pieces: "list[Piece]"

class Parser:

    def from_json(self, data):

        scans = dict((k, self.parse_scan(v)) for k, v in data['scans'].items())
        pieces = [self.parse_piece(p) for p in data['pieces']]
        return Puzzle(scans, pieces)

    def parse_scan(self, data):
        return Puzzle.Scan(data['path'])

    def parse_piece(self, data):
        
        label  = data['label']
        source = self.parse_source(data['source'])

        points = None
        if 'points' in data:
            points = self.parse_points(data['points'])

        tabs = None
        if 'tabs' in data:
            tabs   = [self.parse_tab(i) for i in data['tabs']]

        edges = None
        if 'edges' in data:
            edges  = [self.parse_edge(i) for i in data['edges']]
            
        return Puzzle.Piece(label, source, points, tabs, edges)

    def parse_source(self, data):
        return Puzzle.Piece.Source(data['id'], tuple(data['rect']))

    def parse_points(self, data):
        pts = np.array(puzzler.chain.ChainCode().decode(data))
        pts.setflags(write=0)
        return pts

    def parse_tab(self, data):
        fit_indexes = self.parse_indexes(data['fit_indexes'])
        ellipse = self.parse_ellipse(data['ellipse'])
        indent = data['indent']
        tangent_indexes = self.parse_indexes(data['tangent_indexes'])
        return puzzler.feature.Tab(fit_indexes, ellipse, indent, tangent_indexes)

    def parse_edge(self, data):
        fit_indexes = self.parse_indexes(data['fit_indexes'])
        line = self.parse_line(data['line'])
        return puzzler.feature.Edge(fit_indexes, line)

    def parse_indexes(self, data):
        return tuple(data)

    def parse_ellipse(self, data):

        center = np.array(data['center'])
        semi_major = data['semi_major']
        semi_minor = data['semi_minor']
        phi = data['phi']
        return puzzler.geometry.Ellipse(center, semi_major, semi_minor, phi)

    def parse_line(self, data):

        pt0 = np.array(data['pt0'])
        pt1 = np.array(data['pt1'])
        return puzzler.geometry.Line(np.array((pt0, pt1)))

class Formatter:

    def to_json(self, puzzle):

        scans = dict((k, self.format_scan(v)) for k, v in puzzle.scans.items())
        pieces = [self.format_piece(p) for p in self.sort_by_label(puzzle.pieces)]
        
        return {'scans': scans, 'pieces': pieces}

    def sort_by_label(self, pieces):
        
        vec = []
        for p in pieces:
            m = re.fullmatch(r"^(\w+)(\d+)", p.label)
            k = (m[1], int(m[2])) if m else (p.label, 0)
            vec.append((k, p))
                
        return [p for _, p in sorted(vec)]

    def format_scan(self, s):
        return {'path': s.path}

    def format_piece(self, p):

        ret = {'label': p.label, 'source': self.format_source(p.source) }

        if p.points is not None:
            ret['points'] = self.format_points(p.points)

        if p.tabs is not None:
            ret['tabs'] = [self.format_tab(i) for i in p.tabs]

        if p.edges is not None:
            ret['edges'] = [self.format_edge(i) for i in p.edges]

        return ret

    def format_source(self, s):
        return {'id': s.id, 'rect': list(s.rect)}

    def format_points(self, points):
        return puzzler.chain.ChainCode().encode(points)

    def format_tab(self, tab):
        return {'fit_indexes': self.format_indexes(tab.fit_indexes),
                'ellipse': self.format_ellipse(tab.ellipse),
                'indent': tab.indent,
                'tangent_indexes': self.format_indexes(tab.tangent_indexes)}

    def format_edge(self, edge):
        return {'fit_indexes': self.format_indexes(edge.fit_indexes),
                'line': self.format_line(edge.line)}

    def format_ellipse(self, e):
        return {'center': e.center.tolist(),
                'semi_major': e.semi_major,
                'semi_minor': e.semi_minor,
                'phi': e.phi}

    def format_line(self, l):
        return {'pt0': l.pts[0].tolist(), 'pt1': l.pts[1].tolist()}

    def format_indexes(self, i):
        return list(i)

