import argparse
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import re
import tempfile

class Segmenter:

    def __init__(self, input_dir, output_dir, pad = 50):
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.pad = pad

    def segment_images(self, descriptors):

        for source in descriptors:
            self.segment_image(source)

    def segment_image(self, descriptor):

        image_path = os.path.join(self.input_dir, descriptor['image_path'])
        print(f"{image_path}")
        
        img = cv.imread(image_path)
        assert img is not None
        
        h, w = img.shape[0], img.shape[1]

        s = descriptor['image_scale']
        
        for piece in descriptor['pieces']:
            label = piece['label']
            path = os.path.join(self.output_dir, f"piece_{label}.jpg")
            rx, ry, rw, rh = piece['rect']
            x0 = max(rx * s - self.pad, 0)
            y0 = max(ry * s - self.pad, 0)
            x1 = min((rx + rw) * s + self.pad, w)
            y1 = min((ry + rh) * s + self.pad, h)
            subimage = img[y0:y1,x0:x1]
            subimage = np.rot90(subimage,-1)

            print(f"{path}: {x1-x0} x {y1-y0}")
            cv.imwrite(path, subimage)

class SegmenterUI:

    def __init__(self, input_dir, output_dir, image_path, metadata_path):
        
        self.input_dir     = input_dir
        self.output_dir    = output_dir
        self.image_path    = image_path
        self.metadata_path = metadata_path
        
        self.tempdir  = tempfile.TemporaryDirectory(dir='C:\\Temp')

        img = cv.imread(os.path.join(self.input_dir, self.image_path))
        
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

    def to_json(self):
        
        pieces = [{'rect': r, 'label': l} for r, l in zip(self.rects, self.labels) if l != '']

        return {
            'image_path': self.image_path,
            'image_scale': self.image_scale,
            'image_size': [*self.image_size],
            'pieces':pieces
        }

    def save_json(self):

        data = self.to_json()

        source = self.load_json()
        if source is None:
            source = []
        else:
            
            for i, s in enumerate(source):
                if s['image_path'] == self.image_path:
                    break
            if i < len(source):
                source[i] = data
            else:
                source.append(data)

        with open(self.metadata_path, 'w') as f:
            json.dump(source, f, indent=2)

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

    parser = argparse.ArgumentParser(description="Segment raw piece scans into individual pieces.")
    parser.add_argument("-r", "--root", default=".", help="root directory (default \".\")")
    parser.add_argument("-i", "--input", metavar='INPUT_DIR', default="scans", help="input directory containing images to be segmented (default \"scans\")")
    parser.add_argument("-o", "--output", metavar='OUTPUT_DIR', default="pieces", help="output directory to write segmented pieces (default \"pieces\")" )
    parser.add_argument("-m", "--metadata", default="segment.json", help="metadata file describing segmentation (default \"segment.json\")")
    parser.add_argument("-s", "--segment", action='store_true', help="perform segmentation")
    parser.add_argument("-u", "--ui", action='store_true', help="launch UI")

    args = parser.parse_args()

    input_dir = os.path.join(args.root, args.input)
    output_dir = os.path.join(args.root, args.output)
    metadata_path = os.path.join(args.root, args.metadata)

    if args.segment:
        
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        s = Segmenter(input_dir, output_dir)
        s.segment_images(metadata)
        
    else:
        s = SegmenterUI(input_dir, metadata_path, output_dir)
        s.ui()

if __name__ == '__main__':
    main()
