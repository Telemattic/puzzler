import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
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

    def find_quads_for_pair(self, tabA2, tabB0):

        def prev_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index-1) % n)

        def next_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index+1) % n)

        def normalize(v):
            return sorted((b, a) if a > b else (a, b) for a, b in v)

        def make_raft(v):
            r = self.raftinator
            print(f"make_raft: v={r.format_feature_pairs(v)}")
            return r.make_raft_from_feature_pairs(v)

        for tabA1, tabB1 in [(prev_tab(tabA2), next_tab(tabB0)), (next_tab(tabA2), prev_tab(tabB0))]:

            for tabC3 in self.pairs_for[tabA1]:

                tabC2 = prev_tab(tabC3)
                for tabD0 in self.pairs_for[tabC2]:
                    raft = make_raft([(tabA2, tabB0), (tabA1, tabC3), (tabC2, tabD0)])

                for tabD3 in self.pairs_for[tabB1]:
                    raft = make_raft([(tabA2, tabB0), (tabA1, tabC3), (tabB1, tabD3)])

            for tabD3 in self.pairs_for[tabB1]:

                tabD0 = next_tab(tabD3)
                for tabC2 in self.pairs_for[tabD0]:
                    raft = make_raft([(tabA2, tabB0), (tabB1, tabD3), (tabD0, tabC2)])

def main():
    parser = argparse.ArgumentParser("quadhunter")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-t", "--tabs", required=True)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tabs)

    q = QuadHunter(pieces, tab_pairs)
    q.find_quads_for_pair(Feature('E10','tab',2), Feature('E9','tab',0))

if __name__ == '__main__':
    main()
