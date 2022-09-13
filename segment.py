import argparse
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import re
import tempfile

def describe_type_of(x):

    if isinstance(x, tuple):
        return 'tuple(' + ',\n'.join(describe_type_of(i) for i in x) + ')'
    elif isinstance(x, list):
        return 'list(' + ',\n'.join(describe_type_of(i) for i in x) + ')'
    elif isinstance(x, np.ndarray):
        return f"array[{x.shape}]"
    else:
        return type(x).__name__

class ImageSegmenter:

    def __init__(self, image_path, metadata_path, pieces_path):
        self.image_path    = image_path
        self.metadata_path = metadata_path
        self.pieces_path   = pieces_path
        
        self.tempdir  = tempfile.TemporaryDirectory(dir='C:\\Temp')

        img = cv.imread(self.image_path)
        
        w, h = img.shape[1], img.shape[0]
        self.image_raw = (w, h)

        s = 1
        if w > 1024 or h > 1024:
            s = (max(w,h) + 1023) // 1024
            w //= s
            h //= s

        self.image_size  = (w,h)
        self.image_scale = s

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

        # print(describe_type_of(contours))

        self.contours = contours

        self.rects   = [cv.boundingRect(c) for c in contours]
        print(f"{self.rects=}")

        self.labels  = [''] * len(self.contours)

        metadata = self.load_json()
        if metadata:

            for piece in metadata['pieces']:
                rect = piece['rect']
                x, y = rect[0] + rect[2]//2, rect[1] + rect[3]//2

                for i, r in enumerate(self.rects):
                    if r[0] < x < r[0]+r[2] and r[1] < y < r[1]+r[3]:
                        self.labels[i] = piece['label']

    def do_label(self, xy):

        piece = None

        x, y = xy

        for i, r in enumerate(self.rects):

            x0, y0, w, h = r
            if x0 < x < x0+w and y0 < y < y0+h:
                piece = i
                break

        print(f"{piece=}")
        if piece is None:
            return
        
        self.labels[piece] = self.curr_label()
        self.next_label()
        self.render()

        if self.metadata_path is not None:
            self.save_json()

    def load_json(self):

        if self.metadata_path is None or not os.path.exists(self.metadata_path):
            return None

        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def save_json(self):

        pieces = [{'rect': r, 'label': l} for r, l in zip(self.rects, self.labels)]

        data = {
            'image_path': self.image_path,
            'image_scale': self.image_scale,
            'image_size': [*self.image_size],
            'pieces':pieces
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def do_segment(self):

        img = cv.imread(self.image_path)
        s = self.image_scale
        w, h = img.shape[1], img.shape[0]
        pad  = 50

        for i, l in enumerate(self.labels):
            if l == '':
                continue
            path = os.path.join(self.pieces_path, f"piece_{l}.jpg")
            rx, ry, rw, rh = self.rects[i]
            x0 = max(rx * s - pad, 0)
            y0 = max(ry * s - pad, 0)
            x1 = min((rx + rw) * s + pad, w)
            y1 = min((ry + rh) * s + pad, h)
            subimage = img[y0:y1,x0:x1]

            print(f"{path}: img[{y0}:{y1},{x0}:{x1}]")
            cv.imwrite(path, subimage)

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
            [sg.Text(f"Image: {self.image_path}\nMetadata: {self.metadata_path}\nPieces: {self.pieces_path}")],
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-m", "--metadata")
    parser.add_argument("-p", "--pieces")

    args = parser.parse_args()

    e = ImageSegmenter(args.image, args.metadata, args.pieces)
    e.ui()

if __name__ == '__main__':
    main()
