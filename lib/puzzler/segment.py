import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import re
import tempfile

class Segmenter:

    class Scan:
        
        def __init__(self, path):
            self.path  = path
            self.image = cv.imread(self.path)
            assert(self.image is not None)

        def get_subimage(self, rect, pad=50):

            h, w = self.image.shape[0], self.image.shape[1]
            rx, ry, rw, rh = rect
            x0 = max(rx - pad, 0)
            y0 = max(ry - pad, 0)
            x1 = min(rx + rw + pad, w)
            y1 = min(ry + rh + pad, h)
            subimage = self.image[y0:y1,x0:x1]
            return np.rot90(subimage,-1)

    def __init__(self, output_dir, pad = 50):
        self.output_dir = output_dir
        self.pad = pad

    def segment_images(self, puzzle):

        updated_pieces = []
        for i, s in puzzle['sources'].items():
            pieces = [p for p in puzzle['pieces'] if p['source']['id'] == i]
            updated_pieces += self.segment_image(source, pieces)

        return {'sources': puzzle['sources'], 'pieces': updated_pieces}

    def segment_image(self, source, pieces):

        scan = Scan(source['path'])
        print(f"{scan.path}")

        retval = []
        for piece in pieces:

            if piece is None:
                continue

            label = piece['label']
            path = os.path.join(self.output_dir, f"piece_{label}.jpg")
            img = scan.get_subimage(piece['source']['rect'])

            print(f"{path}: {img.shape[1]} x {img.shape[0]}")
            cv.imwrite(path, img)

            retval.append({'label': label, 'source': piece['source']})

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

        self.max_w = 1200
        self.max_h = 800

        s = 1
        if w > self.max_w or h > self.max_h:
            s = max((w + self.max_w - 1)//self.max_w,
                    (h + self.max_h - 1)//self.max_h)
            w //= s
            h //= s

        self.image_size  = (w,h)
        self.image_scale = s

        # source['scale'] = s
        # source['size'] = [w, h]

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

        self.rects = []
        for c in contours:
            s = self.image_scale
            rx, ry, rw, rh = cv.boundingRect(c)
            self.rects.append((rx*s, ry*s, rw*s, rh*s))
            
        print(f"{self.rects=}")

        self.labels  = [''] * len(self.rects)

        for p in self.puzzle['pieces']:

            if p['source']['id'] != self.source_id:
                continue

            rx, ry, rw, rh = p['source']['rect']
            x, y = rx + rw // 2, ry + rh // 2

            for i, (rx, ry, rw, rh) in enumerate(self.rects):
                if rx < x < rx+rw and ry < y < ry+rh:
                    self.labels[i] = p['label']
            
    def do_label(self, xy):

        piece = None

        x, y = xy
        s = self.image_scale
        x *= s
        y *= s

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

        pieces = [p for p in self.puzzle['pieces'] if p['source']['id'] != self.source_id]
        for rect, label in zip(self.rects, self.labels):
            if label != '':
                source = {'id': self.source_id, 'rect':rect}
                pieces.append({'label': label, 'source': source})

        sources = self.puzzle['sources']
        return {'sources': sources, 'pieces': pieces}

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

        s = self.image_scale
        for rect in self.rects:
            rx, ry, rw, rh = rect
            tl = (rx / s, ry /s)
            br = ((rx + rw)/s, (ry+rh)/s)
            graph.draw_rectangle(tl, br, line_color='yellow')

        for i, l in enumerate(self.labels):
            if l == '':
                continue

            rx, ry, rw, rh = self.rects[i]
            x = (rx + rw // 2) // s
            y = (ry + rh // 2) // s
            graph.draw_text(l, (x,y), font=('Courier', 16, 'bold'), color='yellow')

    def ui(self):

        layout = [
            [sg.Graph(canvas_size=(self.max_w,self.max_h),
                      graph_bottom_left = (0, self.max_h),
                      graph_top_right = (self.max_w, 0),
                      background_color='black',
                      key='graph',
                      enable_events=True,
                      metadata=self)],
            [sg.Text("Label"), sg.InputText("A1", key='label', size=(5,1))]
        ]
        self.window = sg.Window('Image Segmenter', layout, finalize=True)
        self.render()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == 'graph':
                self.do_label(values['graph'])
            else:
                print(event, values)
