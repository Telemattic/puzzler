import json
import numpy as np
import os
import puzzler
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
            
        label: str
        source: Source
        points: np.array

    scans: dict[str, Scan]
    pieces: list[Piece]

    @staticmethod
    def from_json(data):

        scans = dict()
        for k, v in data['sources'].items():
            scans[k] = Puzzle.Scan(v['path'])

        pieces = list()
        for p in data['pieces']:

            label = p['label']
            source = Puzzle.Piece.Source(p['source']['id'], tuple(p['source']['rect']))
            points = puzzler.chain.ChainCode().decode(p['points'])

            pieces.append(Puzzle.Piece(label, source, points))

        return Puzzle(scans, pieces)

    def to_json(puzzle) -> dict:

        scans = dict()
        for k, v in puzzle.scans.items():
            scans[k] = {'path': v.path}

        pieces = list()
        for p in puzzle.pieces:
            source = {'id': p.source.id, 'rect': list(p.source.rect)}
            points = puzzler.chain.ChainCode().encode(p.points)
            pieces.append({'label': p.label, 'source': source, 'points': points})

        return {'sources': scans, 'pieces': pieces}
    
def path_to_id(path):
    id = os.path.splitext(os.path.basename(path))[0]
    return id

def init():
    return {'version':2, 'sources':{}, 'pieces':[]}

def update_0(puzzle):
    source_ids = []
    for s in puzzle['sources']:
        id = path_to_id(s['path'])
        source_ids.append(id)

    for p in puzzle['pieces']:
        p['source']['id'] = source_ids[p['source']['id']]

    sources = dict(zip(source_ids, puzzle['sources']))
    return {'version':1, 'sources':sources, 'pieces':puzzle['pieces']}

def update_1(puzzle):
    
    scales = dict((i, s['scale']) for i, s in puzzle['sources'].items())
    
    for piece in puzzle['pieces']:
        rect = piece['source']['rect']
        scale = scales[piece['source']['id']]
        piece['source']['rect'] = [i * scale for i in rect]

    sources = dict((i, {'path': s['path']}) for i, s in puzzle['sources'].items())

    return {'version': 2, 'sources': sources, 'pieces': puzzle['pieces']}

def update(puzzle):

    version = puzzle.get('version', 0)
    if version == 2:
        return puzzle

    if version == 0:
        puzzle = update_1(update_0(puzzle))

    if version == 1:
        puzzle = update_1(puzzle)

    return puzzle

def load(path):

    with open(path) as f:
        puzzle = json.load(f)

    return update(puzzle)

def save(path, puzzle):

    # assert puzzle['sources'] is dict
    # assert puzzle['pieces'] is list
    with open(path, 'w') as f:
        puzzle['version'] = 2
        json.dump(puzzle, f, indent=2)
