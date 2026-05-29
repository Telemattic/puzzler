import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import csv
import decimal
import itertools
import multiprocessing as mp
import operator
import puzzler
import re
import tqdm

Feature = puzzler.raft.Feature
    
class ScoreComputer:

    def __init__(self, pieces):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(pieces)

    def score_feature_pairs(self, feature_pairs):

        r = self.raftinator
        raft = r.make_raft_from_feature_pairs(feature_pairs)
    
        seams = r.get_seams_for_raft(raft)
        return r.get_total_error_for_raft_and_seams(raft, seams)

    def all_tab_alignments(self, piece_a, piece_b):

        n_tabs = len(piece_a.tabs)
        if n_tabs != len(piece_b.tabs):
            return []

        def make_feature_pairs(offset):
            retval = []
            for i in range(n_tabs):
                tab_a = piece_a.tabs[(i+offset) % n_tabs]
                tab_b = piece_b.tabs[i]
                if tab_a.indent != tab_b.indent:
                    return None
                fa = Feature(piece_a.label, 'tab', (i+offset) % n_tabs)
                fb = Feature(piece_b.label, 'tab', i)
                retval.append((fa, fb))
            return retval
            
        retval = []
        for i in range(n_tabs):
            if x := make_feature_pairs(i):
                retval.append(x)

        return retval

    def single_tab_alignments(self, piece_a, piece_b):

        retval = []
        for i, tab_a in enumerate(piece_a.tabs):
            for j, tab_b in enumerate(piece_b.tabs):
                if tab_a.indent != tab_b.indent:
                    continue
                fa = Feature(piece_a.label, 'tab', i)
                fb = Feature(piece_b.label, 'tab', j)
                retval.append((fa, fb))
        return retval

    def score_pair_align_all_tabs(self, label_a, label_b):

        piece_a = self.pieces[label_a]
        piece_b = self.pieces[label_b]

        n_tabs = len(piece_a.tabs)
        if n_tabs != len(piece_b.tabs):
            return None

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

            score = self.score_feature_pairs(feature_pairs)
            if best_score is None or score < best_score:
                best_score = score

        return best_score

    def score_pair_align_one_tab(self, label_a, label_b):

        piece_a = self.pieces[label_a]
        piece_b = self.pieces[label_b]

        best_score = None

        for i, tab_a in enumerate(piece_a.tabs):
            for j, tab_b in enumerate(piece_b.tabs):
                if tab_a.indent != tab_b.indent:
                    continue

                fp = (Feature(label_a, 'tab', i), Feature(label_b, 'tab', j))
                score = self.score_feature_pairs([fp])
                if best_score is None or score < best_score:
                    best_score = score

        return best_score

    def score_pieces(self, lhs_pieces, rhs_pieces):
    
        rows = []
        for l, r in itertools.product(lhs_pieces, rhs_pieces):
            score = self.score_pair_align_one_tab(l, r)
            if score is not None:
                rows.append({'lhs':l[2:], 'rhs':r[2:], 'score':score})

        return rows

def worker(args, src_q, dst_q):

    lhs = puzzler.file.load(args.left)
    for i in lhs.pieces:
        i.label = 'l_' + i.label
    
    rhs = puzzler.file.load(args.right)
    for i in rhs.pieces:
        i.label = 'r_' + i.label

    score_computer = ScoreComputer({i.label: i for i in (lhs.pieces + rhs.pieces)})
    
    job = src_q.get()
    while job:

        rows = score_computer.score_pieces(*job)

        dst_q.put(rows)
        job = src_q.get()

    return

def iterate_over_puzzles(lhs, rhs):
    lhs = [i.label for i in lhs]
    rhs = [i.label for i in rhs]
    return list(itertools.product(itertools.batched(lhs, 32), itertools.batched(rhs, 32)))
    
def command_score(args):

    lhs = puzzler.file.load(args.left)
    for i in lhs.pieces:
        i.label = 'l_' + i.label
    
    rhs = puzzler.file.load(args.right)
    for i in rhs.pieces:
        i.label = 'r_' + i.label

    with open(args.output, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile, fieldnames='lhs rhs score'.split())
        writer.writeheader()
        
        if args.num_workers:
            src_q = mp.Queue()
            dst_q = mp.Queue()

            workers = [mp.Process(target=worker, args=(args, src_q, dst_q)) for _ in range(args.num_workers)]
            for p in workers:
                p.start()

            num_jobs = 0
            for p in iterate_over_puzzles(lhs.pieces, rhs.pieces):
                src_q.put(p)
                num_jobs += 1
                
            pbar = tqdm.tqdm(total=num_jobs, smoothing=0)
            while num_jobs > 0:
                job = dst_q.get()
                num_jobs -= 1
                pbar.update()
                writer.writerows(job)

            for _ in workers:
                src_q.put(None)

            for p in workers:
                p.join()
                
        else:
            score_computer = ScoreComputer({i.label: i for i in (lhs.pieces + rhs.pieces)})
            for l, r in tqdm.tqdm(iterate_over_puzzles(lhs.pieces, rhs.pieces), smoothing=0):
                writer.writerows(score_computer.score_pieces(l, r))

def command_rank(args):

    with open(args.input, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        fieldnames = reader.fieldnames
        if 'rank' in fieldnames:
            print("match_puzzles: {args.input} already has a rank column?")
            fieldnames.remove(rank)
            
        data = []
        for row in reader:
            row['score'] = float(row['score'])
            data.append(row)

    def nice_sort(s):
        m = re.fullmatch(r"([A-Z]+)(\d+)", s)
        c = int(m[2])
        r = 0
        for i in m[1]:
            r = r*26 + ord(i) - ord('A') + 1
        return (r, c)

    data.sort(key=lambda x: nice_sort(x['lhs']))

    with open(args.output, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile, fieldnames=fieldnames + ['rank'])
        writer.writeheader()
        
        for k, g in itertools.groupby(data, key=operator.itemgetter('lhs')):
            rows = sorted(list(g), key=operator.itemgetter('score'))
            for i, row in enumerate(rows, start=1):
                row['score'] = decimal.Decimal(f"{row['score']:.3f}")
                row['rank'] = i
            writer.writerows(rows)

def command_none(args):
    print("fnord.")
            
def main():
    parser = argparse.ArgumentParser(prog='match_puzzles')
    parser.set_defaults(func=command_none)

    commands = parser.add_subparsers()
    
    parser_score = commands.add_parser("score", help="score the puzzle comparison")
    parser_score.add_argument("-l", "--left", help="left puzzle", required=True)
    parser_score.add_argument("-r", "--right", help="right puzzle", required=True)
    parser_score.add_argument("-n", "--num-workers", help="parallel processing", default=0, type=int)
    parser_score.add_argument("-o", "--output", help="output filename", required=True)
    parser_score.set_defaults(func=command_score)
    
    parser_rank = commands.add_parser("rank", help="rank the already computed scores")
    parser_rank.add_argument("-i", "--input", help="input filename", required=True)
    parser_rank.add_argument("-o", "--output", help="output filename", required=True)
    parser_rank.set_defaults(func=command_rank)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
