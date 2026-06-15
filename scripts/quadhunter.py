import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import functools
import puzzler

import pprint

Feature = puzzler.raft.Feature

class QuadHunter:

    def __init__(self, pieces, tab_pairs):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.tab_pairs = tab_pairs

        self.best_fits = {}
        for p in self.pieces.values():
            for i in range(len(p.tabs)):
                dst = Feature(p.label, 'tab', i)
                src = self.tab_pairs.get_best_fit(dst)
                self.best_fits[dst] = src

        self.pairs_for = collections.defaultdict(set)
        for dst, src in self.best_fits.items():
            self.pairs_for[dst].add(src)
            self.pairs_for[src].add(dst)
            
    def find_quads_for_pair(self, tabA, tabB):

        def normalize(v):
            return sorted((b, a) if a > b else (a, b) for a, b in v)

        retval = []

        r = self.raftinator
        
        candidates = self.find_candidate_quads(tabA, tabB) + self.find_candidate_quads(tabB, tabA)
        for fp in candidates:
            fp = normalize(fp)
            s = r.format_feature_pairs(fp)
            raft = r.make_raft_from_feature_pairs(fp)
            mse = r.get_total_error_for_raft_and_seams(raft)
            retval.append((mse, s))

        return retval

    def find_candidate_quads(self, tabA2, tabB0):

        def prev_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index-1) % n)

        def next_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index+1) % n)

        retval = []

        tabA1 = prev_tab(tabA2)
        tabB1 = next_tab(tabB0)

        for tabC3 in self.pairs_for[tabA1]:

            tabC2 = prev_tab(tabC3)
            for tabD0 in self.pairs_for[tabC2]:
                retval.append([(tabA2, tabB0), (tabA1, tabC3), (tabC2, tabD0)])

            for tabD3 in self.pairs_for[tabB1]:
                retval.append([(tabA2, tabB0), (tabA1, tabC3), (tabB1, tabD3)])

        for tabD3 in self.pairs_for[tabB1]:

            tabD0 = next_tab(tabD3)
            for tabC2 in self.pairs_for[tabD0]:
                retval.append([(tabA2, tabB0), (tabB1, tabD3), (tabD0, tabC2)])

        return retval

def main():
    parser = argparse.ArgumentParser("quadhunter")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-t", "--tabs", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tabs)

    q = QuadHunter(pieces, tab_pairs)

    def test_it(dst, src):
        if not(dst < src and q.best_fits[src] == dst):
            return False
        return len(pieces[dst.piece].tabs) == 4 and len(pieces[src.piece].tabs) == 4

    with open(args.output, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile, fieldnames='dst src raft mse'.split())
        writer.writeheader()
        for dst, src in q.best_fits.items():
            if not test_it(dst, src):
                continue
            
            print(f"{dst!s} <- {src!s}")
            for mse, raft in q.find_quads_for_pair(dst, src):
                writer.writerow({'dst':dst, 'src':src, 'raft':raft, 'mse':f"{mse:.3f}"})
        
    # q.find_quads_for_pair(Feature('E10','tab',2), Feature('E9','tab',0))
        
if __name__ == '__main__':
    main()
