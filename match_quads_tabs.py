import argparse
import csv

def load_quads(input_csv_path):

    quads = []
    with open(input_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in 'col_no', 'row_no', 'rank':
                row[k] = int(row[k])
            for k in 'mse',:
                row[k] = float(row[k])
            quads.append(row)

    return quads

def write_tabs(path, tabs):

    def parse_tab(s):
        a, b = s.split(':')
        return a, int(b)

    fieldnames = 'dst_piece dst_tab_no src_piece src_tab_no'.split()
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        for a, b in tabs.items():
            dst_piece, dst_tab_no = parse_tab(a)
            src_piece, src_tab_no = parse_tab(b)
            writer.writerow({'dst_piece':dst_piece,
                             'dst_tab_no':dst_tab_no,
                             'src_piece':src_piece,
                             'src_tab_no':src_tab_no})

def match_it(quads):

    def parse_raft(s):
        return [tuple(f.split('=')) for f in s.split(',')]

    tabs = dict()
    
    for quad in quads:
        
        if quad['rank'] != 1 or quad['mse'] > 10.:
            continue

        for a, b in parse_raft(quad['raft']):
            
            if a in tabs:
                assert tabs[a] == b
            else:
                tabs[a] = b
                
            if b in tabs:
                assert tabs[b] == a
            else:
                tabs[b] = a

    return tabs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input file of quads data')
    parser.add_argument('-o', '--output', help='output file name for CSV tab correspondence',
                        required=True)

    args = parser.parse_args()

    quads = load_quads(args.filename)
    tabs = match_it(quads)
    write_tabs(args.output, tabs)

if __name__ == '__main__':
    main()
