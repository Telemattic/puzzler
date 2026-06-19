import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import functools
import numpy as np
import puzzler
import tqdm

Feature = puzzler.raft.Feature

def normalize(v):
    # the set insures we remove duplicates which may have been
    # introduced by normalizing feature pair order, etc.
    s = set((b, a) if a > b else (a, b) for a, b in v)
    return tuple(sorted(s))

class QuadHunter:

    def __init__(self, pieces, tab_pairs, use_v2):
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

        self.use_v2 = use_v2
            
    def find_quads_for_pair(self, tabA, tabB):

        retval = []

        r = self.raftinator

        if self.use_v2:
            candidates = self.find_candidate_quads_v2(tabA, tabB)
        else:
            candidates = self.find_candidate_quads_v1(tabA, tabB)
            
        for fp in candidates:
            fp = normalize(fp)
            raft = r.make_raft_from_feature_pairs(fp)
            raft = r.refine_alignment_within_raft(raft)
            fpx = self.compute_tab_matches(raft)
            if not set(fp).issubset(fpx):
                continue
            mse = r.get_total_error_for_raft_and_seams(raft)
            s = r.format_feature_pairs(fpx)
            retval.append((mse, s))

        return retval

    def compute_tab_matches(self, raft):
        fp = puzzler.solver.compute_tab_matches(self.pieces, raft.coords)
        return normalize(fp)

    def find_candidate_quads_v1(self, tabA2, tabB0):

        def prev_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index-1) % n)

        def next_tab(tab):
            n = len(self.pieces[tab.piece].tabs)
            return Feature(tab.piece, 'tab', (tab.index+1) % n)

        def helper(tabA2, tabB0):

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

        return helper(tabX, tabY) + helper(tabY, tabX)

    def find_candidate_quads_v2(self, tabX, tabY):

        verbose = False

        def get_tab_uv(p, i):
            t = p.tabs[i]
            v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
            v = v / np.linalg.norm(v, axis=1)
            v = np.sum(v, axis=0)
            v = v / np.linalg.norm(v)
            if not t.indent:
                v = -v
            if verbose:
                with np.printoptions(precision=3):
                    print(f"  get_tab_uv: p={p.label} {i=} {v=}")
            return v

        def cross2d(x, y):
            z = x[...,0]*y[...,1] - x[...,1] * y[...,0]
            if verbose:
                print(f"cross2d: {z=:.3f}")
            return z

        def prev_tab(tab):
            p = self.pieces[tab.piece]
            i = tab.index
            j = (tab.index - 1) % len(p.tabs)
            if verbose:
                print(f"prev_tab: {tab=!s} {j=}")
            if cross2d(get_tab_uv(p,i), get_tab_uv(p,j)) < .8:
                return None
            return Feature(tab.piece, 'tab', j)

        def next_tab(tab):
            p = self.pieces[tab.piece]
            i = tab.index
            j = (tab.index + 1) % len(p.tabs)
            if verbose:
                print(f"next_tab: {tab=!s} {j=}")
            if cross2d(get_tab_uv(p,j), get_tab_uv(p,i)) < .8:
                return None
            return Feature(tab.piece, 'tab', j)

        raft = self.get_raft(normalize([(tabX, tabY)]))

        # print(f"{tabX=!s} {prev_tab(tabX)=!s}")
        # print(f"{tabX=!s} {next_tab(tabX)=!s}")
        
        # print(f"{tabY=!s} {prev_tab(tabY)=!s}")
        # print(f"{tabY=!s} {next_tab(tabY)=!s}")

        retval = []
        for tabA1, tabB1 in self.get_straightline_pairs(raft, verbose):

            if verbose:
                print(f"{tabA1=!s} {tabB1=!s}")

            for tabC3 in self.pairs_for[tabA1]:

                if verbose:
                    print(f"  {tabC3=!s}")

                tabC2 = prev_tab(tabC3)
                if verbose:
                    print(f"  {tabC2=!s}")

                for tabD0 in self.pairs_for[tabC2]:
                    if verbose:
                        print(f"    {tabD0=!s}")
                    retval.append([(tabX, tabY), (tabA1, tabC3), (tabC2, tabD0)])

                for tabD3 in self.pairs_for[tabB1]:
                    if verbose:
                        print(f"    {tabD3=!s}")
                    retval.append([(tabX, tabY), (tabA1, tabC3), (tabB1, tabD3)])

            for tabD3 in self.pairs_for[tabB1]:

                if verbose:
                    print(f"  {tabD3=!s}")

                tabD0 = next_tab(tabD3)
                if verbose:
                    print(f"  {tabD0=!s}")

                for tabC2 in self.pairs_for[tabD0]:
                    if verbose:
                        print(f"    {tabC2=!s}")
                    retval.append([(tabX, tabY), (tabB1, tabD3), (tabD0, tabC2)])

        return retval

    def get_straightline_pairs(self, raft, verbose=False):
        if verbose:
            print("get_straightline_pairs:")
        retval = []
        tabs = self.get_detailed_tabs(raft)
        for i, curr in enumerate(tabs):
            prev = tabs[i-1]
            dot_product = np.sum(prev.tab_uv * curr.tab_uv)
            if verbose:
                with np.printoptions(precision=3):
                    print(f"  {prev=}")
                    print(f"  {curr=}")
                    print(f"  {dot_product=}")
            if dot_product > .9:
                retval.append((Feature(prev.piece,'tab',prev.index), Feature(curr.piece, 'tab', curr.index)))
        if verbose:
            print(f"  {retval=}")
        return retval

    @functools.cache
    def get_raft(self, fp):

        r = self.raftinator
        return r.make_raft_from_feature_pairs(fp)

    def get_detailed_tabs(self, raft):

        r = self.raftinator

        closest_pieces = puzzler.frontier.ClosestPieces(
            self.pieces, raft.coords, None, r.factory.distance_query_cache)
        adjacency = dict((i, closest_pieces(i)) for i in raft.coords)
    
        bc = puzzler.frontier.BoundaryComputer(self.pieces)
        frontiers = bc.find_boundaries_from_adjacency(adjacency)
        assert len(frontiers)==1
        
        fe = puzzler.frontier.FrontierExplorer(self.pieces, raft.coords)
        return fe.get_detailed_tabs(frontiers[0])

def get_seed_pairs(best_fits):
    retval = []
    for dst, src in best_fits.items():
        if dst < src and best_fits[src] == dst:
            retval.append((dst, src))
    return sorted(retval)

def main():
    parser = argparse.ArgumentParser("quadhunter")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-t", "--tabs", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--version", type=int, default=2)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tabs)

    print(f"Using algorithm v{args.version}")
    q = QuadHunter(pieces, tab_pairs, args.version == 2)

    with open(args.output, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile, fieldnames='dst src raft mse'.split())
        writer.writeheader()

        for dst, src in tqdm.tqdm(get_seed_pairs(q.best_fits)):
            for mse, raft in q.find_quads_for_pair(dst, src):
                writer.writerow({'dst':dst, 'src':src, 'raft':raft, 'mse':f"{mse:.3f}"})
        
if __name__ == '__main__':
    main()
