import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import csv
import puzzler

def get_rows(tab_pairs, dst_tab, src_tabs):

    ranks = {}
    try:
        for rank in range(1, len(src_tabs)+1):
            src_tab = tab_pairs.get_ranked_fit(dst_tab, rank)
            ranks[src_tab] = rank
    except ValueError:
        pass
    
    retval = []
    for src_tab in src_tabs:
        fit_error = tab_pairs.get_fit_error(dst_tab, src_tab)
        retval.append({'dst_label':dst_tab.piece, 'dst_tab_no':dst_tab.index,
                       'src_label':src_tab.piece, 'src_tab_no':src_tab.index,
                       'sse':fit_error.sse, 'n':fit_error.n,
                       'rank':ranks.get(src_tab)})

    return retval

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--puzzle', required=True)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)

    outdents = []
    indents = []
    for p in puzzle.pieces:
        for tab_no, tab in enumerate(p.tabs):
            f = puzzler.raft.Feature(p.label, 'tab', tab_no)
            if tab.indent:
                indents.append(f)
            else:
                outdents.append(f)

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.input)

    with open(args.output, 'w', newline='') as f:
        
        writer = csv.DictWriter(f, fieldnames='dst_label dst_tab_no src_label src_tab_no sse n rank'.split())
        writer.writeheader()

        for dst_tab in outdents:
            writer.writerows(get_rows(tab_pairs, dst_tab, indents))

        for dst_tab in indents:
            writer.writerows(get_rows(tab_pairs, dst_tab, outdents))

if __name__ == '__main__':
    main()
