import argparse
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import re
import tempfile

class Segmenter:

    def __init__(self, output_dir, pad = 50):
        self.output_dir = output_dir
        self.pad = pad

    def segment_images(self, puzzle):

        updated_pieces = []
        for i, source in enumerate(puzzle['sources']):
            pieces = Segmenter.find_pieces_for_source(puzzle, i)
            updated_pieces += self.segment_image(source, pieces)

        return {'sources': puzzle['sources'], 'pieces': updated_pieces}

    @staticmethod
    def find_pieces_for_source(puzzle, source_id):

        rects = puzzle['sources'][source_id]['rects']
        pieces = [None] * len(rects)

        for piece in puzzle['pieces']:

            i, x, y = piece['source']
            if i != source_id:
                continue

            for j, rect in enumerate(rects):
                rx, ry, rw, rh = rect
                if rx < x < rx+rw and ry < y < ry+rh:
                    x, y = rx + rw//2, ry + rh//2
                    pieces[j] = {'label': piece['label'], 'source':[i, x, y]}
                    break

        return pieces

    def segment_image(self, source, pieces):

        image_path = os.path.join(source['path'])
        print(f"{image_path}")
        
        img = cv.imread(image_path)
        assert img is not None
        
        h, w = img.shape[0], img.shape[1]

        s = source['scale']

        retval = []
        for rect, piece in zip(source['rects'], pieces):

            if piece is None:
                continue

            label = piece['label']
            path = os.path.join(self.output_dir, f"piece_{label}.jpg")
            rx, ry, rw, rh = rect
            x0 = max(rx * s - self.pad, 0)
            y0 = max(ry * s - self.pad, 0)
            x1 = min((rx + rw) * s + self.pad, w)
            y1 = min((ry + rh) * s + self.pad, h)
            subimage = img[y0:y1,x0:x1]
            subimage = np.rot90(subimage,-1)

            print(f"{path}: {x1-x0} x {y1-y0}")
            cv.imwrite(path, subimage)

            i, x, y = piece['source']
            x = rx + rw // 2
            y = ry + rh // 2
            retval.append({'label': label, 'source': [i, x, y]})

        return retval

class SegmenterUI:

    def __init__(self, puzzle, source_id):

        self.puzzle    = puzzle
        self.source_id = source_id

        self.tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp')

        source = self.puzzle['sources'][source_id]
        
        img = cv.imread(source['path'])
        
        w, h = img.shape[1], img.shape[0]
        self.image_raw = (w, h)

        s = 1
        if w > 1024 or h > 1024:
            s = (max(w,h) + 1023) // 1024
            w //= s
            h //= s

        self.image_size  = (w,h)
        self.image_scale = s

        source['scale'] = s
        source['size'] = [w, h]

        print(f"image={self.image_raw}")
        print(f"scale={self.image_scale} -> {self.image_size}")
        
        img      = cv.resize(img, self.image_size, cv.INTER_CUBIC)
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh   = 255 - cv.threshold(gray, 84, 255, cv.THRESH_BINARY_INV)[1]
        dilate   = cv.dilate(thresh, cv.getStructuringElement(cv.MORPH_RECT, (2,2)))

        print(f"temp={self.tempdir.name}")

        cv.imwrite(os.path.join(self.tempdir.name, "color.png"),  img)
        cv.imwrite(os.path.join(self.tempdir.name, "gray.png"),   gray)
        cv.imwrite(os.path.join(self.tempdir.name, "thresh.png"), thresh)
        cv.imwrite(os.path.join(self.tempdir.name, "dilate.png"), dilate)
        
        contours = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        assert len(contours)==2

        contours = [c for c in contours[0] if cv.contourArea(c) > 2000]
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            print(f"{i}: area={area:.1f}")

        self.contours = contours

        self.rects   = [cv.boundingRect(c) for c in contours]
        print(f"{self.rects=}")

        self.labels  = [''] * len(self.rects)

        pieces = Segmenter.find_pieces_for_source(self.puzzle, self.source_id)
        for i, p in enumerate(pieces):
            if p is not None:
                self.labels[i] = p['label']

    def do_label(self, xy):

        piece = None

        x, y = xy

        for i, rect in enumerate(self.rects):

            rx, ry, rw, rh = rect
            if rx < x < rx+rw and ry < y < ry+rh:
                piece = i
                break

        print(f"{piece=}")
        if piece is None:
            return
        
        self.labels[piece] = self.curr_label()
        self.next_label()
        self.render()

    def to_json(self):

        pieces = [p for p in self.puzzle['pieces'] if p['source'][0] != self.source_id]
        for rect, label in zip(self.rects, self.labels):
            if label != '':
                rx, ry, rw, rh = rect
                x, y = rx + rw // 2, ry + rh // 2
                pieces.append({'label': label, 'source': [self.source_id, x, y]})

        sources = self.puzzle['sources']
        sources[self.source_id]['rects'] = self.rects
        return {'sources': sources, 'pieces':pieces}

    def do_segment(self):

        segmenter = Segmenter(self.input_dir, self.output_dir)
        segmenter.segment_image(self.to_json())

    def curr_label(self):
        return self.window['label'].get()

    def next_label(self):
        m = re.fullmatch("(\w)(\d+)", self.curr_label())
        if m is None:
            return
        
        prefix = m[1]
        suffix = int(m[2])
        self.window['label'].update(prefix + str(suffix+1))

    def render(self):

        graph = self.window['graph']
        graph.erase()

        path = os.path.join(self.tempdir.name, "color.png")
        graph.draw_image(filename=path, location=(0,0))

        for contour in self.contours:
            points = [tuple(xy) for xy in np.squeeze(contour)]
            graph.draw_lines(points, color='red', width=2)

        for r in self.rects:
            tl = (r[0], r[1])
            br = (r[0]+r[2], r[1]+r[3])
            graph.draw_rectangle(tl, br, line_color='yellow')

        for i, l in enumerate(self.labels):
            if l == '':
                continue

            r = self.rects[i]
            x = r[0] + r[2] // 2
            y = r[1] + r[3] // 2
            graph.draw_text(l, (x,y), color='yellow')

    def ui(self):

        layout = [
            [sg.Graph(canvas_size=(1024,1024),
                      graph_bottom_left = (0,1023),
                      graph_top_right = (1023,0),
                      background_color='black',
                      key='graph',
                      enable_events=True,
                      metadata=self)],
            [sg.Text("Label"), sg.InputText("A1", key='label', size=(5,1))],
            [sg.Button('Segment', key='button_segment')]
        ]
        self.window = sg.Window('Image Segmenter', layout, finalize=True)
        self.render()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == 'graph':
                self.do_label(values['graph'])
            elif event == 'button_segment':
                self.do_segment()
            else:
                print(event, values)
