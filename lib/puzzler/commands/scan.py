import puzzler

def scan_add(args):

    puzzle = puzzler.file.load(args.puzzle)

    id = puzzler.file.path_to_id(args.image)
    source = {'path': args.image}
    puzzle['sources'][id] = source

    s = puzzler.segment.SegmenterUI(puzzle, id)
    s.ui()

    puzzler.file.save(args.puzzle, s.to_json())

def scan_edit(args):

    puzzle = puzzler.file.load(args.puzzle)

    s = puzzler.segment.SegmenterUI(puzzle, args.id)
    s.ui()

    puzzler.file.save(args.puzzle, s.to_json())
        
def scan_list(args):

    puzzle = puzzler.file.load(args.puzzle)

    for i, s in puzzle['sources'].items():
        print(f"{i}: {s['path']}")

def scan_output(args):

    puzzle = puzzler.file.load(args.puzzle)

    s = puzzler.segment.Segmenter(args.output)
    s.segment_images(puzzle)

def add_parser(commands):
    parser_scan = commands.add_parser("scan", help="label pieces in scans")
    
    commands = parser_scan.add_subparsers()

    parser_add = commands.add_parser("add", help="add an image")
    parser_add.add_argument("image", help="path to image")
    parser_add.set_defaults(func=scan_add)

    parser_edit = commands.add_parser("edit", help="edit an image")
    parser_edit.add_argument("id", help="id of image to edit")
    parser_edit.set_defaults(func=scan_edit)

    parser_list = commands.add_parser("list", help="list images")
    parser_list.set_defaults(func=scan_list)

    parser_output = commands.add_parser("output", help="output labeled pieces")
    parser_output.add_argument("output", metavar="DIRECTORY")
    parser_output.set_defaults(func=scan_output)
