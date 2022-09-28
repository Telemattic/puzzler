import json
import numpy as np
import os
import puzzler
from dataclasses import dataclass
    
def path_to_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def init():
    return puzzler.puzzle.Puzzle(dict(), list())

def load(path):

    with open(path) as f:
        puzzle = json.load(f)

    return puzzler.puzzle.Parser().from_json(puzzle)

def save(path, puzzle):

    data = puzzler.puzzle.Formatter().to_json(puzzle)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
