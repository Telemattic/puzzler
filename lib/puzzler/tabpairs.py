import csv
import functools
import itertools
import json
import mmap
import operator
import puzzler
import re
import struct
import logging

logger = logging.getLogger('puzzler')

Feature = puzzler.raft.Feature
FitError = puzzler.raft.FitError

class TabPairsMMap:

    MAGIC = 0xcafebeef

    class OffsetComputer:

        def __init__(self, o):
            self.offset = o['offset']
            self.length = o['length']
            self.n_rows = o['n_rows']
            self.n_cols = o['n_cols']
            self.row_size = o['row_size']
            self.elem_size = o['elem_size']

        def __call__(self, row_no, col_no):
            return self.offset + row_no * self.row_size + col_no * self.elem_size

    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), length=0, access=mmap.ACCESS_READ)

        magic, json_length, json_offset = struct.unpack_from('=LLQ', self.mm, offset=0)
        if magic != TabPairsMMap.MAGIC:
            raise ValueError(f"TabRanksMMap: bad magic number {magic:X}")

        s = self.mm[json_offset:json_offset+json_length].decode()
        o = json.loads(s)

        # each tab (indent and outdent combined) has a unique integer id
        self.id_to_tab = [self.parse_tab(i) for i in o['id_to_tab']]
        self.tab_to_id = {j:i for i, j in enumerate(self.id_to_tab)}

        # each tab maps to a unique dense index among its siblings
        # (indents or outdents)
        self.n_rows = o['n_rows']
        self.tab_to_row = {self.parse_tab(i):j for i, j in o['tab_to_row'].items()}

        self.n_cols = o['n_cols']
        self.tab_to_col = {self.parse_tab(i):j for i, j in o['tab_to_col'].items()}

        self.fit_offset_computer = TabPairsMMap.OffsetComputer(o['fit_table'])
        self.rank_offset_computer = TabPairsMMap.OffsetComputer(o['rank_table'])
        
    def get_fit_error(self, dst_tab, src_tab):
        dst_row = self.tab_to_row[dst_tab]
        src_col = self.tab_to_col[src_tab]
        o = self.fit_offset_computer(dst_row, src_col)
        sse, n = struct.unpack_from("=fH", self.mm, o)
        return FitError(sse, n)

    def get_ranked_fit(self, dst_tab, rank_no):
        if not (1 <= rank_no <= self.n_cols):
            raise ValueError(f"rank={rank_no}, must be between 1 and {self.n_cols}")
        dst_row = self.tab_to_row[dst_tab]
        src_col = rank_no-1
        o = self.rank_offset_computer(dst_row, src_col)
        src_id, = struct.unpack_from("=H", self.mm, o)
        if src_id == 0xFFFF:
            raise ValueError(f"no ranked fit for tab={dst_tab!s} rank={rank_no}")
        return self.id_to_tab[src_id]

    @staticmethod
    def parse_tab(s):
        label, tab_no = s.split(':')
        return Feature(label, 'tab', int(tab_no))

def load_tab_pairs(path):

    if re.search(r"\.mmap$", path):
        return TabPairsMMap(path)
    raise Exception(f"load_tab_pairs: don't know how to parse {path}")

def read_tab_pairs_csv(csv_path):

    retval = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst_tab = Feature(row['dst_label'], 'tab', int(row['dst_tab_no']))
            src_tab = Feature(row['src_label'], 'tab', int(row['src_tab_no']))
            sse = float(row['sse'])
            n = int(row['n'])
            retval.append((dst_tab, src_tab, sse, n))

    return retval

def write_tab_pairs_csv_to_mmap(mmap_opath, csv_ipath, outdent_tabs, indent_tabs):

    # the tables are immediately after the header
    file_length = 16
    
    def allocate_block(n):
        nonlocal file_length
        # double alignment
        o = (file_length + 7) & ~7
        file_length = o + n
        return o

    def allocate_table(n_rows, n_cols, elem_size):
        length = n_rows * n_cols * elem_size
        offset = allocate_block(length)
        return {'offset':offset, 'length':length,
                'n_rows':n_rows, 'n_cols':n_cols,
                'row_size':n_cols * elem_size,
                'elem_size':elem_size}

    def fill_table(mm, table, cell_value):
        
        init_data = b'\xFF\xFF' * table['n_cols']
        offset = table['offset']
        length = table['length']
        row_size = table['row_size']
        for i in range(offset, offset+length, row_size):
            mm[i:i+row_size] = init_data

    outdent_tabs = sorted(outdent_tabs)
    indent_tabs = sorted(indent_tabs)

    id_to_tab = sorted(outdent_tabs + indent_tabs)
    tab_to_row = {j:i for i, j in enumerate(id_to_tab)}
    assert len(tab_to_row) == len(id_to_tab)

    # it's the same encoding, this just makes the semantics more
    # obvious
    tab_to_id = tab_to_row

    tab_to_col = {j:i for i, j in enumerate(outdent_tabs)} | {j:i for i, j in enumerate(indent_tabs)}
    assert len(tab_to_col) == len(id_to_tab)

    n_rows = len(id_to_tab)
    n_cols = max(len(outdent_tabs), len(indent_tabs))

    fit_table = allocate_table(n_rows, n_cols, 6)
    rank_table = allocate_table(n_rows, n_cols, 2)

    o = {
        'fit_table': fit_table,
        'rank_table': rank_table,
        'id_to_tab': [str(i) if i else i for i in id_to_tab],
        'n_rows': n_rows,
        'tab_to_row': {str(i):j for i, j in tab_to_row.items()},
        'n_cols': n_cols,
        'tab_to_col': {str(i):j for i, j in tab_to_col.items()}
    }

    json_bytes = json.dumps(o).encode()

    json_length = len(json_bytes)
    json_offset = allocate_block(json_length)

    fit_offset_computer = TabPairsMMap.OffsetComputer(fit_table)
    rank_offset_computer = TabPairsMMap.OffsetComputer(rank_table)

    csv_data = read_tab_pairs_csv(csv_ipath)

    # 0: dst_tab
    # 1: src_tab
    # 2: SSE
    # 3: n

    # group by dst_tab
    group_key = operator.itemgetter(0)
    
    csv_data.sort(key=group_key)

    # within a group sort by MSE=SSE/n and break ties by srt_tab
    sort_key = lambda x: (x[2]/x[3], x[1])

    with open(mmap_opath, 'w+b') as ofile:
        mm = mmap.mmap(ofile.fileno(), file_length)
        struct.pack_into('=LLQ', mm, 0, TabPairsMMap.MAGIC, json_length, json_offset)
        mm[json_offset:json_offset+json_length] = json_bytes

        fill_table(mm, rank_table, b'\xFF\xFF')

        for k, g in itertools.groupby(csv_data, key=group_key):

            rows = list(g)
            rows.sort(key=sort_key)

            for rank, (dst_tab, src_tab, sse, n) in enumerate(rows, start=1):

                dst_row = tab_to_row[dst_tab]
                src_col = tab_to_col[src_tab]
                o = fit_offset_computer(dst_row, src_col)
                struct.pack_into('=fH', mm, o, sse, n)

                src_id = tab_to_id[src_tab]
                o = rank_offset_computer(dst_row, rank-1)
                struct.pack_into('=H', mm, o, src_id)
