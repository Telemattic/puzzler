import json

def init():
    return {'sources':[], 'pieces':[]}

def load(path):

    with open(path) as f:
        data = json.load(f)

    return data

def save(path, puzzle):

    with open(path, 'w') as f:
        json.dump(puzzle, f, indent=2)
