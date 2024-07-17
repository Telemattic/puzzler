import argparse
import collections
import json
import os
import subprocess
import tempfile

def number_nodes(expected_pairs, everybody_else=None):
    
    node_ids = dict()
    curr = 'A1'
    i = 0
    while curr not in node_ids:
        node_ids[curr] = i
        i += 1
        curr = expected_pairs[curr]

    if everybody_else:
        for k, v in everybody_else.items():
            if k not in node_ids:
                node_ids[k] = i
                i += 1
            if v not in node_ids:
                node_ids[v] = i
                i += 1

    return node_ids

def get_good_nodes(expected, actual):

    def helper(A, B):
        # a good node has the same input for both A and B
        return set(k for k, v in A.items() if v == B.get(k, '*'))

    def invert(A):
        B = dict()
        for k, v in A.items():
            if v in B:
                B[v] = B[v] + ',' + k
            else:
                B[v] = k
        return B

    return helper(expected, actual) & helper(invert(expected), invert(actual))

def merge_good_runs(good_nodes, node_ids):

    all_nodes = sorted(node_ids.keys(), key=lambda x: node_ids[x])

    i = 0
    last_node = len(all_nodes)-1
    if False and all_nodes[i] in good_nodes:
        j = last_node
        while j > 0 and all_nodes[j] in good_nodes:
            node_ids[all_nodes[j]] = node_ids[all_nodes[i]]
            j -= 1
        last_node = j

    i = 0
    while i <= last_node:
        if all_nodes[i] in good_nodes:
            j = i+1
            while j <= last_node and all_nodes[j] in good_nodes:
                node_ids[all_nodes[j]] = node_ids[all_nodes[i]]
                j += 1
            i = j
        i += 1

    return node_ids

def make_graph(ofile, expected, actual, layout='linear'):

    # three kinds of edge: good, bad, missing

    in_edges = collections.defaultdict(list)
    for dst, src in actual.items():
        expected_src = expected.get(dst)
        if expected_src is None:
            in_edges[dst].append((src, 'bad'))
        elif expected_src == src:
            in_edges[dst].append((src, 'good'))
        else:
            in_edges[dst].append((src, 'bad'))
            in_edges[dst].append((expected_src, 'missing'))

    out_edges = collections.defaultdict(list)
    for dst, edges in in_edges.items():
        for src, kind in edges:
            out_edges[src].append((dst, kind))

    node_to_id = number_nodes(expected, actual)
    id_to_nodes = dict((v, set([k])) for k, v in node_to_id.items())
    ordered_nodes = sorted(node_to_id.keys(), key=lambda x: node_to_id[x])

    # for node in ordered_nodes:
    #     print(f"{node=} in_edges={in_edges[node]} out_edges={out_edges[node]}")
    
    def has_single_good_edge(dst, src):
        return (len(out_edges[src]) == 1 and
                len(in_edges[dst]) == 1 and
                out_edges[src][0] == (dst, 'good') and
                in_edges[dst][0] == (src, 'good'))

    def merge_nodes(a, b):

        new_id = node_to_id[a]
        old_id = node_to_id[b]

        assert old_id != new_id
        
        B = id_to_nodes.pop(old_id)
        for x in B:
            node_to_id[x] = new_id

        id_to_nodes[new_id] |= B

    def single_good_parent(node):
        if len(in_edges[node]) != 1:
            return None
        parent, kind = in_edges[node][0]
        if kind != 'good':
            return None
        return parent

    def order_by_connectivity(v):

        s = set(v)

        head = v[0]
        parent = single_good_parent(head)
        while parent in s:
            head = parent
            parent = single_good_parent(head)

        ret = list()
        while head in s:
            if head in ret:
                assert len(s) == len(ret)
                break
            ret.append(head)
            head = out_edges[head][0][0]

        return ret[::-1]

    ordered_expected_nodes = sorted(expected.keys(), key=lambda x: node_to_id[x])

    for i in range(len(ordered_expected_nodes)):
        a = ordered_expected_nodes[i-1]
        b = ordered_expected_nodes[i]
        # print(f"merge_nodes: test {a},{b}")
        if has_single_good_edge(a, b):
            merge_nodes(a, b)

    # print(f"{id_to_nodes=}")

    labels = collections.defaultdict(list)
    for k in ordered_nodes:
        labels[node_to_id[k]].append(k)

    for k, v in labels.items():
        if len(v) > 1:
            v = order_by_connectivity(v)
            labels[k] = v[-1] + '..' + v[0]
        else:
            labels[k] = v[0]

    f = ofile

    print('digraph G {', file=f)

    if layout == 'circular':
        print('  layout="twopi";', file=f)
        print('  ranksep=4;', file=f)
        print('  root=CENTER;', file=f)
        print('  edge [style=invis];', file=f)
        print('  CENTER [style=invis];', file=f)
        print('  CENTER -> {', file=f)
        
        for i in sorted(id_to_nodes.keys()):
            print(f'    {i}', file=f)
            
        print('  }', file=f)
        
    print('  edge [style=solid];', file=f)
    print('  node [shape=rectangle];', file=f)

    for k, v in labels.items():
        print(f'   {k} [label="{v}"];', file=f)

    for src, dsts in out_edges.items():
        src_id = node_to_id[src]
        for dst, kind in dsts:
            dst_id = node_to_id[dst]
            if src_id == dst_id and kind == 'good':
                continue
                    
            # print(f"{kind} edge from {src} to {dst}, but these nodes have been merged?")
            
            if kind == 'good':
                print(f'  {src_id} -> {dst_id} [arrowhead=lvee];', file=f)
            elif kind == 'bad':
                print(f'  {src_id} -> {dst_id} [color=red, arrowhead=rvee];', file=f)
            else:
                print(f'  {src_id} -> {dst_id} [style=dashed, arrowhead=lvee];', file=f)
                
    print('}', file=f)

def make_pdf(dotpath, pdfpath):
    # "C:\Program Files\Graphviz\bin\dot.exe" -Tpdf -Gsize=8,10.5 -Gpage=8.5,11 -o <pdf_path> <dot_path>
    exe = r'C:\Program Files\Graphviz\bin\dot.exe'
    args = [exe, '-Tpdf', '-Gsize=8,10.5', '-Gpage=8.5,11', '-o', pdfpath, dotpath]
    print(' '.join(args))
    subprocess.run(args)

def show_pdf(pdfpath):
    subprocess.run(['cmd', '/c', 'start', pdfpath])

def main():

    parser = argparse.ArgumentParser(
        prog='GraphIt',
        description='Graph piece connectivity data')
    parser.add_argument('filename', help='input json file containing connectivity data')
    parser.add_argument('-c', '--circular', help='circular layout', dest='layout',
                        action='store_const', const='circular')
    parser.add_argument('-l', '--linear', help='linear layout (default)', dest='layout',
                        action='store_const', const='linear')
    parser.add_argument('-o', '--output', help='output filename (default: None)')

    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        s = f.read()
        o = json.loads(s)

    if args.output.endswith('.pdf'):

        # delete_on_close depends on python 3.12
        with tempfile.NamedTemporaryFile(dir=r'C:\Temp', prefix='graphit_', suffix='.dot',
                                         mode='w', newline='\n', delete=False) as dotfile:
            make_graph(dotfile, o['expected_pairs'], o['actual_pairs'], args.layout)
            dotfile.close()
            make_pdf(dotfile.name, args.output)
            os.unlink(dotfile.name)

    elif args.output.endswith('.dot'):
            
        make_graph(args.output, o['expected_pairs'], o['actual_pairs'], args.layout)

if __name__ == '__main__':
    main()
