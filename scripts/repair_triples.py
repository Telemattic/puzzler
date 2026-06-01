import argparse
import csv
import decimal
import re

def load_tabpairs(path):

    retval = dict()
    with open(path, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        fieldnames = reader.fieldnames
        
        if 'fwd_sse' in fieldnames and 'fwd_n' in fieldnames:
            sse = 'fwd_sse'
            n = 'fwd_n'
        else:
            sse = 'sse'
            n = 'n'

        for row in reader:
            dst = (row['dst_label'], 'tab', int(row['dst_tab_no']))
            src = (row['src_label'], 'tab', int(row['src_tab_no']))
            fit_error = (float(row[sse]), int(row[n]))
            retval[dst,src] = fit_error

    return retval

def load_expected(path):
    good_matches = set()

    with open(path, 'r', newline='') as f:
        
        reader = csv.DictReader(f)
        for row in reader:
            dst = (row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = (row['src_piece'], 'tab', int(row['src_tab_no']))
            good_matches.add((dst, src))

    return good_matches

def load_quads(path):
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        quads = [row for row in reader if int(row['rank']) == 1]

    # we simply processed them and assigned quad_no in the order
    # they were found in the file
    oldmap = dict()
    for i, row in enumerate(quads):
        r = int(row['row_no'])
        c = int(row['col_no'])
        oldmap[(r,c)] = 4*i
 
    # make the new numbering deterministic, using the (row_no,
    # col_no) tuple to order the quads
    remap = dict()
    for i, (rc, old_quad_no) in enumerate(sorted(oldmap.items())):
        new_quad_no = i*4
        for j in range(4):
            remap[old_quad_no+j] = new_quad_no+j

    return remap

def parse_feature(f):
    m = re.match("([A-Z]+[0-9]+):([0-9])", f)
    assert m
    return (m[1], 'tab', int(m[2]))

def parse_raft(raft):
    retval = []
    for fp in raft.split(','):
        a, b = fp.split('=',2)
        retval.append((parse_feature(a), parse_feature(b)))
    return retval

def repair_row(row, tabpairs, expected, quads):

    def compute_lower_bound_mse(raft, fit_piece):
        sse = 0.
        n = 0
        for fp in parse_raft(raft):
            if fp[1][0] == fit_piece:
                fit_error = tabpairs[fp]
                sse += fit_error[0]
                n += fit_error[1]

        mse = None
        if n > 0:
            mse = sse / n
            mse = decimal.Decimal(f"{mse:.3f}")

        return mse

    def is_correct_fit(raft):
        return all(fp in expected for fp in parse_raft(raft))

    if tabpairs:
        row['lower_bound_mse'] = compute_lower_bound_mse(row['raft'], row['fit_piece'])
        
    if expected:
        row['correct_fit'] = 1 if is_correct_fit(row['raft']) else 0

    if quads:
        row['quad_no'] = quads[int(row['quad_no'])]
        
    return row

def repair_triples(ipath, opath, tabpairs, expected, quads):

    with open(ipath, 'r', newline='') as ifile:
        with open(opath, 'w', newline='') as ofile:

            reader = csv.DictReader(ifile)
            
            fieldnames=reader.fieldnames
            if 'correct_fit' not in fieldnames:
                fieldnames.append('correct_fit')
            
            writer = csv.DictWriter(ofile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                writer.writerow(repair_row(row, tabpairs, expected, quads))
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input csv file of triples', required=True)
    parser.add_argument('-o', '--output', help='output csv file of "repaired" triples', required=True)
    parser.add_argument('-t', '--tabpairs', help='input csv file of tabpairs, updates lower_bound_mse')
    parser.add_argument('-e', '--expected', help='input csv of good tab matches, updates correct_fit')
    parser.add_argument('-q', '--quads', help='input csv of quads used for original triples execution, updates quad_no')

    args = parser.parse_args()

    tabpairs = None
    if args.tabpairs:
        tabpairs = load_tabpairs(args.tabpairs)

    expected = None
    if args.expected:
        expected = load_expected(args.expected)

    quads = None
    if args.quads:
        quads = load_quads(args.quads)

    repair_triples(args.input, args.output, tabpairs, expected, quads)
     
if __name__ == '__main__':
    main()
