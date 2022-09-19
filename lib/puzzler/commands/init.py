import puzzler

def puzzle_init(args):
    puzzle = puzzler.file.init()
    puzzler.file.save(args.puzzle, puzzle)
    
def add_parser(commands):
    parser_init = commands.add_parser("init", help="initialize an empty puzzle")
    parser_init.set_defaults(func=puzzle_init)
