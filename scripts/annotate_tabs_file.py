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

def annotate_tabs_file(good_matches, input_path, output_path):

    ifile = open(input_path, 'r', newline='')
    reader = csv.DictReader(ifile)
            
    ofile = open(output_path, 'w', newline='')
    skip_fields = {'src_coord_x', 'src_coord_y', 'src_coord_angle', 'src_index_0', 'src_index_1', 'neighbor'}
    ofields = [i for i in reader.fieldnames if i not in skip_fields] + ['good_match']
            
    writer = csv.DictWriter(ofile, fieldnames=ofields, extrasaction='ignore')
    writer.writeheader()

    for row in reader:

        key = tuple(row[i] for i in ('dst_label', 'dst_tab_no', 'src_label', 'src_tab_no'))
        mse = float(row['mse'])
        row['mse'] = decimal.Decimal(f"{mse:.3f}")
        row['good_match'] = key in good_matches

        writer.writerow(row)

    ofile.close()
    ifile.close()

def parse_constraint(s):
    if m := re.fullmatch("(\w+):(\d+)=(\w+):(\d+)", s):
        return (m[1], m[2], m[3], m[4])
    elif re.fullmatch("(\w+)/(\d+)=(\w+)/(\d+)", s):
        # ignore edge constraints
        return None
    else:
        raise ValueError(f"bad constraint: {s}")

def parse_raft(s):
    retval = []
    for i in s.split(','):
        x = parse_constraint(i)
        if x is not None:
            retval.append(x)
    return retval
    return [parse_constraint(i) for i in s.split(',')]

def annotate_raft_file(good_matches, input_path, output_path):

    def is_good_raft(raft):
        return all(i in good_matches for i in parse_raft(raft))

    dialect = get_dialect(input_path)
    
    ifile = open(input_path, 'r', newline='')
    reader = csv.DictReader(ifile, dialect=dialect)
            
    ofile = open(output_path, 'w', newline='')
    ofields = reader.fieldnames.copy()
    
    ofields.remove('mse')
    for i in range(4):
        ofields.append(f'mse{i}')

    add_quad_no = 'quad_no' not in ofields
    if add_quad_no:
        ofields.append('quad_no')

    add_init_rank = 'init_rank' not in ofields
    if add_init_rank:
        ofields.append('init_rank')

    ofields.remove('rank')
    ofields.append('rank')
    ofields.append('good_match')

    def update_rows(rows):

        for r in rows:
            for i, mse in enumerate(r.pop('mse').split(',')):
                r[f'mse{i}'] = mse

        if add_init_rank:
            vals = [(float(r['mse0']), i) for i, r in enumerate(rows)]
            for i, (_, row_no) in enumerate(sorted(vals), start=1):
                rows[row_no]['init_rank'] = i
            
        if add_quad_no:
            for r in rows:
                r['quad_no'] = quad_no

        for r in rows:
            r['good_match'] = is_good_raft(r['raft'])

        return rows
        
    writer = csv.DictWriter(ofile, fieldnames=ofields, extrasaction='ignore')
    writer.writeheader()

    quad_no = -1
    rows = []
    for row in reader:
        
        if row['rank'] == '1':
            writer.writerows(update_rows(rows))
            quad_no += 1
            rows = []

        rows.append(row)

    writer.writerows(update_rows(rows))

    ofile.close()
    ifile.close()

def get_dialect(path):

    with open(path, 'r', newline='') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))

    return dialect

def detect_input_file_type(path):

    dialect = get_dialect(path)
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f, dialect=dialect)
        fn = set(reader.fieldnames)

    if 'raft' in fn:
        return 'raft'

    if all(i in fn for i in ['dst_label', 'dst_tab_no', 'src_label', 'src_tab_no']):
        return 'tabs'

    return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ranked', help='input csv file of ranked tab matches', required=True)
    parser.add_argument('-g', '--good', help='input csv file of good (known correct) tab matches', required=True)
    parser.add_argument('-o', '--output', help='output csv file of annotated ranked matches', required=True)

    args = parser.parse_args()

    file_type = detect_input_file_type(args.ranked)
    if file_type is None:
        print(f"{args.ranked}: file format not recognized")
        return

    good_matches = load_good_matches_as_set(args.good)
    if file_type == 'tabs':
        annotate_tabs_file(good_matches, args.ranked, args.output)
    elif file_type == 'raft':
        annotate_raft_file(good_matches, args.ranked, args.output)
     
if __name__ == '__main__':
    main()
