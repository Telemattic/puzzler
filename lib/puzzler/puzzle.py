import puzzler

import numpy as np
from dataclasses import dataclass

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
            rect: tuple[int, int, int, int]
            
        label:  str
        source: Source
        points: np.array
        tabs:   list[puzzler.feature.Tab]
        edges:  list[puzzler.feature.Edge]

    scans: dict[str, Scan]
    pieces: list[Piece]

class Parser:

    def from_json(self, data):

        scans = dict()
        for k, v in data['sources'].items():
            scans[k] = self.parse_scan(v)

        pieces = [self.parse_piece(p) for pieces in data['pieces']]

        return Puzzle(scans, pieces)

    def parse_scan(self, data):
        return Puzzle.Scan(data['path'])

    def parse_piece(self, data):
        
        label  = data['label']
        source = self.parse_source(data['source'])
        points = self.parse_points(data['points'])
        tabs   = [self.parse_tab(i) for i in data['tabs']]
        edges  = [self.parse_edge(i) for i in data['edges']]
        return Puzzle.Piece(label, source, points, tabs, edges)

    def parse_source(self, data):
        return Puzzle.Piece.Source(data['id'], tuple(data['rect']))

    def parse_points(self, data):
        return puzzler.chain.ChainCode().decode(data)

    def parse_tab(self, data):
        fit_indexes = self.parse_indexes(data['fit_indexes'])
        ellipse = self.parse_ellipse(data['ellipse'])
        indent = data['indent']
        tangent_indexes = self.parse_indexes(data['tangent_indexes'])
        return Tab(fit_indexes, ellipse, indent, tangent_indexes)

    def parse_edge(self, data):
        fit_indexes = self.parse_indexes(data['fit_indexes'])
        line = self.parse_line(data['line'])

    def parse_indexes(self, data):
        return tuple(data)

    def parse_ellipse(self, data):

        center = np.array(data['center'])
        semi_major = data['semi_major']
        semi_minor = data['semi_minor']
        phi = data['phi']
        return Ellipse(center, semi_major, semi_minor, phi)

    def parse_line(self, data):

        pt0 = np.array(data['pt0'])
        pt1 = np.array(data['pt1'])
        return Line(pt0, pt1)

class Formatter:

    def to_json(puzzle):

        scans = dict()
        for k, v in puzzle.scans.items():
            scans[k] = format_scan(v)

        pieces = [self.format_piece(p) for p in puzzle.pieces]

        return {'scans': scans, 'pieces': pieces}

    def format_scan(s):
        return {'path': s.path}

    def format_source(s):
        return {'id': s.id, 'rect': list(s.rect)}

    def format_points(points):
        return puzzler.chain.ChainCode().encode(points)

    def format_piece(p):
        
        source = self.format_source(p.source)
        points = self.format_points(p.points)
        tabs   = [self.format_tab(i) for i in p.tabs]
        edges  = [self.format_edge(i) for i in p.edges]

        return {'label': p.label, 'source': scans, 'points': points, 'tabs': tabs, 'edges': edges}

    def format_tab(tab):

        return {'fit_indexes': self.format_indexes(tab.fit_indexes),
                'ellipse': self.format_ellipse(tab.ellipse),
                'indent': tab.indent,
                'tangent_indexes': self.format_indexes(tab.tangent_indexes)}

    def format_edge(edge):
        
        return {'fit_indexes': self.format_indexes(edge.fit_indexes),
                'line': self.format_line(edge.line)}

    def format_ellipse(e):

        return {'center': e.center.tolist(),
                'semi_major': e.semi_major,
                'semi_minor': e.semi_minor,
                'phi': e.phi}

    def format_line(l):

        return {'pt0': l.pt0.tolist(), 'pt1': l.pt1.tolist()}

    def format_indexes(self, i):
                
        return list(i)

