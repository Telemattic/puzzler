import csv
import decimal
import argparse
import re

def load_good_matches_as_set(path):

    good_matches = set()

    with open(path, 'r', newline='') as f:
        
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[i] for i in ('dst_piece', 'dst_tab_no', 'src_piece', 'src_tab_no'))
            good_matches.add(key)

    return good_matches

def parse_constraint(s):
    m = re.fullmatch("(\w+):(\d+)=(\w+):(\d+)", s)
    if m is None:
        raise ValueError(f"bad constraint: {s}")

    return (m[1], m[2], m[3], m[4])

def parse_raft(s):
    return [parse_constraint(i) for i in s.split(',')]

def error_tracks(good_matches, input_path, output_path):
    
    def is_good_raft(raft):
        return all(i in good_matches for i in parse_raft(raft))
    
    def process_rows(input_rows):

        if len(input_rows) == 0:
            return []

        output_rows = []

        for iter_no, key in enumerate(('mse0', 'mse1', 'mse2', 'mse3')):

            vals = [(float(irow[key]), row_no) for row_no, irow in enumerate(input_rows)]
            for rank, (mse, row_no) in enumerate(sorted(vals), start=1):

                irow = input_rows[row_no]
                orow = {
                    'mse': mse,
                    'rank': rank,
                    'quad_no': irow['quad_no'],
                    'raft': irow['raft'],
                    'good_match': is_good_raft(irow['raft']),
                    'iter_no': iter_no,
                }
                output_rows.append(orow)

        return output_rows

    ifile = open(input_path, 'r', newline='')
    reader = csv.DictReader(ifile)

    ofile = open(output_path, 'w', newline='')
    writer = csv.DictWriter(ofile, fieldnames='quad_no raft rank good_match iter_no mse'.split())
    writer.writeheader()

    quad_no = ''
    rows = []
    
    for irow in reader:

        if irow['quad_no'] != quad_no:
            writer.writerows(process_rows(rows))
            quad_no = irow['quad_no']
            rows = []

        rows.append(irow)
        
    writer.writerows(process_rows(rows))

    ofile.close()
    ifile.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ranked', help='input csv file of ranked tab matches', required=True)
    parser.add_argument('-g', '--good', help='input csv file of good (known correct) tab matches', required=True)
    parser.add_argument('-o', '--output', help='output csv file of annotated ranked matches', required=True)

    args = parser.parse_args()

    good_matches = load_good_matches_as_set(args.good)
    error_tracks(good_matches, args.ranked, args.output)

if __name__ == '__main__':
    main()
