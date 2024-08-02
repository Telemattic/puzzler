import argparse
import csv
import operator

def sort_quads(ipath, opath):

    print(f"{ipath} -> {opath}")

    fieldnames = None
    rows = []
    
    with open(ipath, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        fieldnames = reader.fieldnames.copy()
        rows = [row for row in reader]

    rows.sort(key=lambda x: int(x['quad_no']))

    with open(opath, 'w', newline='') as ofile:
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv file of unsorted quads')
    parser.add_argument('-o', '--output', help='output csv file of sorted quads', required=True)

    args = parser.parse_args()

    sort_quads(args.input, args.output)
     
if __name__ == '__main__':
    main()
