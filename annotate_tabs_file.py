import csv
import decimal
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ranked', help='input csv file of ranked tab matches', required=True)
    parser.add_argument('-g', '--good', help='input csv file of good (known correct) tab matches', required=True)
    parser.add_argument('-o', '--output', help='output csv file of annotated ranked matches', required=True)

    args = parser.parse_args()

    good_matches = set()

    with open(args.good, 'r', newline='') as goodf:
        
        reader = csv.DictReader(goodf)

        for row in reader:
            
            key = tuple(row[i] for i in ('dst_piece', 'dst_tab_no', 'src_piece', 'src_tab_no'))
            good_matches.add(key)

    with open(args.ranked, 'r', newline='') as rankedf:
        
        reader = csv.DictReader(rankedf)
            
        with open(args.output, 'w', newline='') as outputf:
            
            skip_fields = {'src_coord_x', 'src_coord_y', 'src_coord_angle', 'src_index_0', 'src_index_1', 'neighbor'}
            ofields = [i for i in reader.fieldnames if i not in skip_fields] + ['good_match']
            
            writer = csv.DictWriter(outputf, fieldnames=ofields, extrasaction='ignore')
            writer.writeheader()

            for row in reader:

                key = tuple(row[i] for i in ('dst_label', 'dst_tab_no', 'src_label', 'src_tab_no'))
                mse = float(row['mse'])
                row['mse'] = decimal.Decimal(f"{mse:.3f}")
                row['good_match'] = key in good_matches

                writer.writerow(row)
        
if __name__ == '__main__':
    main()
