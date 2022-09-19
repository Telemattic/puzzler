import argparse
import base64
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import tempfile
import zlib

class ChainCode:

    def encode(self, contour):

        code = {(1,0): '0', (1,1): '1', (0,1): '2', (-1,1): '3',
                (-1,0): '4', (-1,-1): '5', (0,-1): '6', (1,-1): '7'}

        path = []
        prev = contour[0]
        for xy in contour[1:]:
            dxdy = (xy[0]-prev[0], xy[1]-prev[1])
            path.append(code[dxdy])
            prev = xy

        raw_data = bytes(''.join(path), 'utf-8')
        cmp_data = zlib.compress(raw_data)
        b64_data = base64.standard_b64encode(cmp_data)

        start = contour[0].tolist()
        chain = str(b64_data, encoding='utf-8')

        return (start, chain)

class PerimeterComputer:

    def __init__(self, image_path, save_images = False):
        
        self.tempdir = None
        if save_images:
            self.tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp')

        img = cv.imread(image_path)
        assert img is not None
        
        w, h = img.shape[1], img.shape[0]
        self.image_size  = (w,h)
        self.images = []

        print(f"image={image_path} {w}x{h}")
        
        img      = cv.resize(img, self.image_size, cv.INTER_CUBIC)
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gray     = cv.medianBlur(gray, 7)
        gray     = cv.GaussianBlur(gray, (3,3), 0)
        
        thresh   = 255 - cv.threshold(gray, 60, 255, cv.THRESH_BINARY_INV)[1]
        # thresh   = cv.medianBlur(thresh, 29)

        self._add_temp_image("color.png", img, 'Source')
        self._add_temp_image("gray.png", gray, 'Gray')
        self._add_temp_image("thresh.png", thresh, 'Thresh')
        
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        self.contour = max(contours[0], key=cv.contourArea)

    def to_json(self):
        start, chain = ChainCode().encode(np.squeeze(self.contour))
        return {'start':start, 'chain':chain}

    def write_json(self, path):

        data = self.to_json()
        with open(path, 'w') as f:
            json.dump(data, f)

    def _add_temp_image(self, filename, img, label):

        if self.tempdir is None:
            return
        
        path = os.path.join(self.tempdir.name, filename)
        cv.imwrite(path, img)
        self.images.append((label, path))

class PerimeterUI:

    def __init__(self, image_path, output_path = None):

        self.pc = PerimeterComputer(image_path, True)

        if output_path:
            self.pc.write_json(output_path)

    def get_image_path(self):

        for label, path in self.pc.images:
            if self.window['radio_image_' + label].get():
                return path
        return None

    def draw_histogram(self):

        graph = self.window['graph']
        
        img = cv.imread(self.get_image_path())
        bgr = len(img.shape) == 3 and img.shape[2]==3

        if bgr:
            for i, color in enumerate(['blue', 'green', 'red']):
                hist = cv.calcHist([img], [i], None, [256], [0, 256])
                scale = 100. / np.max(hist)
                points = [(10+i, 110-scale*v) for i, v in enumerate(np.squeeze(hist))]
                graph.draw_lines(points, color=color)
        else:
            hist = cv.calcHist([img], [0], None, [256], [0, 256])
            scale = 100. / np.max(hist)
            points = [(10+i, 110-scale*v) for i, v in enumerate(np.squeeze(hist))]
            graph.draw_lines(points, color='black')

    def draw_traces(self):
        
        graph = self.window['graph']
        
        img = cv.imread(self.get_image_path())
        bgr = len(img.shape) == 3 and img.shape[2]==3

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        if bgr:
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        if bgr:
            for i, color in enumerate(['blue', 'green', 'red']):
                horiz = [(x,cy-v) for x, v in enumerate(img[cy,:,i])]
                graph.draw_lines(horiz, color=color)
        else:
            horiz = [(x,cy-v) for x, v in enumerate(img[cy,:])]
            graph.draw_lines(horiz, color='black')
            
        graph.draw_line((0,cy), (w-1,cy))
        
        if bgr:
            for i, color in enumerate(['blue', 'green', 'red']):
                vert = [(cx+v,y) for y, v in enumerate(img[:,cx,i])]
                graph.draw_lines(vert, color=color)
        else:
            vert = [(cx+v,y) for y, v in enumerate(img[:,cx])]
            graph.draw_lines(vert, color='black')
            
        graph.draw_line((cx,0), (cx,h-1))

    def draw_contour(self):
        graph = self.window['graph']
        
        points = [tuple(xy) for xy in np.squeeze(self.pc.contour)]
        graph.draw_lines(points, color='red', width=2)

    def render(self):

        graph = self.window['graph']
        graph.erase()

        path = self.get_image_path()
        graph.draw_image(filename=path, location=(0,0))

        if self.window['render_traces'].get():
            self.draw_traces()

        if self.window['render_histogram'].get():
            self.draw_histogram()

        if self.window['render_contour'].get():
            self.draw_contour()

    def _init_ui(self):
        
        w, h = self.pc.image_size

        controls = []
        
        for image in self.pc.images:
            label = image[0]
            is_default = 0 == len(controls)
            key = 'radio_image_' + label
            r = sg.Radio(label, 'radio_image', default=is_default, enable_events=True, key=key)
            controls.append(r)

        controls.append(sg.CB('Render Histogram', default=False, enable_events=True, key='render_histogram'))
        controls.append(sg.CB('Render Traces', default=False, enable_events=True, key='render_traces'))
        controls.append(sg.CB('Render Contour', default=True, enable_events=True, key='render_contour'))
        
        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h),
                      graph_top_right = (w, 0),
                      background_color='black',
                      key='graph',
                      enable_events=True),
             controls],
        ]
        
        self.window = sg.Window('Perimeter Computer', layout, finalize=True)
        self.render()

    def run(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event.startswith('radio_image_'):
                self.render()
            elif event.startswith('render_'):
                self.render()
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--image")
    parser.add_argument("-o", "--output")
    parser.add_argument("-b", "--batch")

    args = parser.parse_args()

    if args.batch:

        with open(args.batch) as f:
            data = json.load(f)

        out = []
        for piece in data:

            pc = PerimeterComputer(piece['image_path'])
            ret = pc.to_json()
            ret['label'] = piece['label']

            out.append(ret)

        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)

    else:
        ui = PerimeterUI(args.image, args.output)
        ui.run()

if __name__ == '__main__':
    main()
