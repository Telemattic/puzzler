import csv
import json
import mmap
import puzzler
import struct

Feature = puzzler.raft.Feature
FitError = puzzler.raft.FitError

class TabPairsMMap:

    MAGIC = 0xcafebabe

    class OffsetComputer:

        def __init__(self, outdent_table_offset, indent_table_offset, outdent_tabs, indent_tabs):
            self.outdent_table_offset = outdent_table_offset
            self.indent_table_offset = indent_table_offset
            self.outdent_tabs = dict((k, v) for v, k in enumerate(outdent_tabs))
            self.indent_tabs = dict((k, v) for v, k in enumerate(indent_tabs))

        def __call__(self, dst_tab, src_tab):
            dst_idx = self.indent_tabs.get(dst_tab)
            if dst_idx is None:
                offset = self.outdent_table_offset
                dst_idx = self.outdent_tabs[dst_tab]
                src_idx = self.indent_tabs[src_tab]
                stride = len(self.indent_tabs)
            else:
                offset = self.indent_table_offset
                src_idx = self.outdent_tabs[src_tab]
                stride = len(self.outdent_tabs)

            return offset + (dst_idx * stride + src_idx) * 6
            
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.MAP_DENYWRITE)

        magic, length, offset = struct.unpack_from('LLQ', self.mm, offset=0)
        assert magic == TabPairsMMap.MAGIC

        s = self.mm[offset:offset+length].decode()
        o = json.loads(s)

        self.offset_computer = TabPairsMMap.OffsetComputer(
            o['outdent_table_offset'],
            o['indent_table_offset'],
            [self.parse_tab(i) for i in o['outdent_tabs']],
            [self.parse_tab(i) for i in o['indent_tabs']])
        
    def get_fit_error(self, dst_tab, src_tab):
        o = self.offset_computer(dst_tab, src_tab)
        sse, n = struct.unpack_from("fH", self.mm, o)
        return FitError(sse, n)

    @staticmethod
    def parse_tab(s):
        label, tab_no = s.split(':')
        return Feature(label, 'tab', int(tab_no))

class TabPairsCSV:

    def __init__(self, path):
        self.data = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(ifile)
            for row in reader:
                dst_tab = Feature(row['dst_label'], 'tab', int(row['dst_tab_no']))
                src_tab = Feature(row['src_label'], 'tab', int(row['src_tab_no']))
                err_sse = float(row['sse'])
                err_n = int(row['n'])
                self.data[(dst_tab,src_tab)] = FitError(err_sse, errr_n)

    def get_fit_error(self, dst_tab, src_tab):
        return self.data[(dst_tab,src_tab)]

def write_tab_pairs_csv_to_mmap(mmap_opath, csv_ipath, outdent_tabs, indent_tabs):

    outdent_tabs = sorted(outdent_tabs)
    indent_tabs = sorted(indent_tabs)

    table_size = len(outdent_tabs) * len(indent_tabs) * 6

    # the tables are immediately after the header
    outdent_table_offset = 16

    # indent table is 16B aligned
    indent_table_offset = (outdent_table_offset + table_size + 15) & ~15

    o = {'outdent_table_offset': outdent_table_offset,
         'indent_table_offset': indent_table_offset,
         'outdent_tabs': [str(i) for i in outdent_tabs],
         'indent_tabs': [str(i) for i in indent_tabs]}

    json_bytes = json.dumps(o).encode()

    json_offset = (indent_table_offset + table_size + 15) & ~15
    json_length = len(json_bytes)

    file_length = json_offset + json_length

    offset_computer = TabPairsMMap.OffsetComputer(
        outdent_table_offset, indent_table_offset, outdent_tabs, indent_tabs)

    with open(mmap_opath, 'w+b') as ofile:
        mm = mmap.mmap(ofile.fileno(), file_length)
        struct.pack_into('LLQ', mm, 0, TabPairsMMap.MAGIC, json_length, json_offset)
        mm[json_offset:json_offset+json_length] = json_bytes

        with open(csv_ipath, 'r', newline='') as ifile:
            reader = csv.DictReader(ifile)
            for row in reader:
                dst_tab = Feature(row['dst_label'], 'tab', int(row['dst_tab_no']))
                src_tab = Feature(row['src_label'], 'tab', int(row['src_tab_no']))
                err_sse = float(row['sse'])
                err_n = int(row['n'])

                o = offset_computer(dst_tab, src_tab)
                struct.pack_into('fH', mm, o, err_sse, err_n)

    
