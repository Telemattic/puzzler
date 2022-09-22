import json
import os

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
