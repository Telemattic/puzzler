import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import puzzler

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

    puzzler.tabpairs.write_tab_pairs_csv_to_mmap(args.output, args.input, outdents, indents)

if __name__ == '__main__':
    main()
