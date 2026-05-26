import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import inspect
import json
import pprint
import puzzler

class SpeedscopeTool:

    def __init__(self):
        pass

    def process_frame(self, f):
        pass

def fnord():
    pprint.pp(inspect.getmembers(puzzler.raft.Raftinator.align_and_merge_rafts_with_feature_pairs, inspect.iscode))
    
def main():
    fnord()

if __name__ == '__main__':
    main()
