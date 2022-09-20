import argparse
import os
import sys

def main():

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
    import puzzler
    
    parser = argparse.ArgumentParser(description="PuZzLeR")

    parser.add_argument("-p", "--puzzle", help="puzzle file", required=True)
    
    commands = parser.add_subparsers()

    puzzler.commands.init.add_parser(commands)
    puzzler.commands.scan.add_parser(commands)
    puzzler.commands.points.add_parser(commands)
    puzzler.commands.ellipse.add_parser(commands)
    puzzler.commands.align.add_parser(commands)

    args = parser.parse_args()
    args.func(args)

    exit(0)

if __name__ == '__main__':
    main()
