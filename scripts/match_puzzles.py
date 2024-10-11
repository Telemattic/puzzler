import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import csv
import operator
import puzzler
from tqdm import tqdm

def score_pair2(piece_a, piece_b, fp):
    
    r = puzzler.raft.Raftinator({piece_a.label:piece_a, piece_b.label:piece_b})
    raft = r.make_raft_from_feature_pairs(fp)
    
    seams = r.get_seams_for_raft(raft)
    mse = r.get_cumulative_error_for_seams(seams)
    mse2 = r.get_total_error_for_raft_and_seams(raft, seams)
    
    # print(f"{r.format_feature_pairs(fp)}: MSE={mse:.3f} MSE2={mse2:.3f}")

    return mse2

def score_pair(piece_a, piece_b):

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

        score = score_pair2(piece_a, piece_b, feature_pairs)
        if best_score is None or score < best_score:
            best_score = score
    
    return best_score

def score_pair_alt(piece_a, piece_b):

    Feature = puzzler.raft.Feature
    
    label_a = piece_a.label
    label_b = piece_b.label

    best_score = None

    for i, tab_a in enumerate(piece_a.tabs):
        for j, tab_b in enumerate(piece_b.tabs):
            if tab_a.indent != tab_b.indent:
                continue

            fp = (Feature(label_a, 'tab', i), Feature(label_b, 'tab', j))
            score = score_pair2(piece_a, piece_b, [fp])
            if best_score is None or score < best_score:
                best_score = score
    
    return best_score

def score_puzzles(lhs, rhs, ofile):
    
    no_matches = []
    
    writer = csv.DictWriter(ofile, fieldnames='lhs rhs rank score'.split())
    writer.writeheader()

    for l in tqdm(lhs.pieces):

        rows = []
        for r in rhs.pieces:
            score = score_pair_alt(l, r)
            if score is not None:
                rows.append({'lhs':l.label, 'rhs':r.label, 'rank':None, 'score':score})

        if not rows:
            no_matches.append(l.label)

        rows.sort(key=operator.itemgetter('score'))
        for rank, row in enumerate(rows, start=1):
            row['lhs'] = row['lhs'][2:]
            row['rhs'] = row['rhs'][2:]
            row['rank'] = rank

        writer.writerows(rows)

    if no_matches:
        print(f"LHS pieces with no matches in RHS: {', '.join(no_matches)}")

def main():
    parser = argparse.ArgumentParser(prog='match_puzzles')
    parser.add_argument("-l", "--left", help="left puzzle", required=True)
    parser.add_argument("-r", "--right", help="right puzzle", required=True)
    parser.add_argument("-o", "--output", help="output filename", required=True)
    args = parser.parse_args()

    lhs = puzzler.file.load(args.left)
    for i in lhs.pieces:
        i.label = 'l_' + i.label
    
    rhs = puzzler.file.load(args.right)
    for i in rhs.pieces:
        i.label = 'r_' + i.label

    with open(args.output, 'w', newline='') as ofile:
        score_puzzles(lhs, rhs, ofile)

if __name__ == '__main__':
    main()
