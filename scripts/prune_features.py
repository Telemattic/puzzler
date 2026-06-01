import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import puzzler

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--puzzle', required=True,
                        help='input puzzle')
    parser.add_argument('features', nargs='+')

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}
    r = puzzler.raft.Raftinator(pieces)

    for i in args.features:
        f = r.parse_feature(i)
        if f.kind == 'tab':
            pieces[f.piece].tabs.pop(f.index)
        elif f.kind == 'edge':
            pieces[f.piece].edges.pop(f.index)

    puzzler.file.save(args.puzzle, puzzle)

if __name__ == '__main__':
    main()
