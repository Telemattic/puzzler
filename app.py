import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import puzzler

def segment_add(args):

    puzzle = puzzler.file.load(args.puzzle)

    id = len(puzzle['sources'])
    source = {'path': args.image, 'rects':[]}
    puzzle['sources'].append(source)

    s = puzzler.segment.SegmenterUI(puzzle, id)
    s.ui()

    puzzler.file.save(args.puzzle, puzzle)

def segment_edit(args):

    puzzle = puzzler.file.load(args.puzzle)

    s = puzzler.segment.SegmenterUI(puzzle, args.id)
    s.ui()

    puzzler.file.save(args.puzzle, puzzle)
        
def segment_list(args):

    puzzle = puzzler.file.load(args.puzzle)

    for i, s in enumerate(puzzle['sources']):
        print(f"{i}: {s['path']}")

def segment_output(args):

    puzzle = puzzler.file.load(args.puzzle)

    s = puzzler.segment.Segmenter(args.output)
    s.segment_images(puzzle)

def puzzle_init(args):

    puzzle = puzzler.file.init()
    puzzler.file.save(args.puzzle, puzzle)

def main():

    parser = argparse.ArgumentParser(description="PuZzLeR")

    parser.add_argument("-p", "--puzzle", help="puzzle file", required=True)
    
    commands = parser.add_subparsers()

    puzzler.commands.init.add_parser(commands)
    puzzler.commands.scan.add_parser(commands)

    args = parser.parse_args()
    args.func(args)

    exit(0)
    
    parser_init = commands.add_parser("init", help="initialize an empty puzzle")
    parser_init.set_defaults(func=puzzle_init)
    
    parser_segment = commands.add_parser("segment", help="label pieces in scans")
    
    commands = parser_segment.add_subparsers()

    parser_add = commands.add_parser("add", help="add an image")
    parser_add.add_argument("image", help="path to image")
    parser_add.set_defaults(func=segment_add)

    parser_edit = commands.add_parser("edit", help="edit an image")
    parser_edit.add_argument("id", help="id of image to edit", type=int)
    parser_edit.set_defaults(func=segment_edit)

    parser_list = commands.add_parser("list", help="list images")
    parser_list.set_defaults(func=segment_list)

    parser_output = commands.add_parser("output", help="output labeled pieces")
    parser_output.add_argument("output", metavar="DIRECTORY")
    parser_output.set_defaults(func=segment_output)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
