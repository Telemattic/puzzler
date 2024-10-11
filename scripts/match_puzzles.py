import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import csv
import operator
import puzzler
from tqdm import tqdm

def score_match2(piece_a, piece_b, fp):
    
    r = puzzler.raft.Raftinator({piece_a.label:piece_a, piece_b.label:piece_b})
    raft = r.make_raft_from_feature_pairs(fp)
    
    seams = r.get_seams_for_raft(raft)
    mse = r.get_cumulative_error_for_seams(seams)
    mse2 = r.get_total_error_for_raft_and_seams(raft, seams)
    
    # print(f"{r.format_feature_pairs(fp)}: MSE={mse:.3f} MSE2={mse2:.3f}")

    return mse2

def score_match(piece_a, piece_b):

    Feature = puzzler.raft.Feature
    
    n_tabs = len(piece_a.tabs)
    if n_tabs != len(piece_b.tabs):
        return None

    label_a = piece_a.label
    label_b = piece_b.label

    best_score = None

    for j in range(n_tabs):
        
        tab_mismatch = False
        feature_pairs = []
        for i in range(n_tabs):
            feature_pairs.append((Feature(label_a, 'tab', (i+j) % n_tabs), Feature(label_b, 'tab', i)))
            tab_a = piece_a.tabs[(i+j) % n_tabs]
            tab_b = piece_b.tabs[i % n_tabs]
            if tab_a.indent != tab_b.indent:
                tab_mismatch = True
                
        if tab_mismatch:
            continue

        score = score_match2(piece_a, piece_b, feature_pairs)
        if best_score is None or score < best_score:
            best_score = score
    
    return best_score

def match_puzzles(puzzle_a, puzzle_b, ofile):
    
    writer = csv.DictWriter(ofile, fieldnames='lhs rhs rank score'.split())
    writer.writeheader()
    
    pieces_a = puzzle_a.pieces
    pieces_b = puzzle_b.pieces

    for a in tqdm(pieces_a):

        rows = []
        for b in pieces_b:
            score = score_match(a, b)
            if score is not None:
                rows.append({'lhs':a.label, 'rhs':b.label, 'rank':None, 'score':score})

        rows.sort(key=operator.itemgetter('score'))
        for rank, row in enumerate(rows, start=1):
            row['rank'] = rank

        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(prog='match_puzzles')
    parser.add_argument("puzzle_a")
    parser.add_argument("puzzle_b")
    parser.add_argument("-o", "--output", help="output filename", required=True)
    args = parser.parse_args()

    a = puzzler.file.load(args.puzzle_a)
    for i in a.pieces:
        i.label = 'l_' + i.label
    
    b = puzzler.file.load(args.puzzle_b)
    for i in b.pieces:
        i.label = 'r_' + i.label

    with open(args.output, 'w', newline='') as ofile:
        match_puzzles(a, b, ofile)

if __name__ == '__main__':
    main()
