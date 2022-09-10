import argparse
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import tempfile

class PerimeterComputer:

    def __init__(self, image_path, output_path = None):
        
        self.image_path    = image_path
        
        self.tempdir  = tempfile.TemporaryDirectory(dir='C:\\Temp')

        img = cv.imread(self.image_path)
        
        w, h = img.shape[1], img.shape[0]
        self.image_size  = (w,h)

        print(f"image={self.image_path} {w}x{h}")
        
        img      = cv.resize(img, self.image_size, cv.INTER_CUBIC)
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh   = 255 - cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]

        print(f"temp={self.tempdir.name}")

        self.images = []
        self._add_image(os.path.join(self.tempdir.name, "color.png"), img, 'Source')
        self._add_image(os.path.join(self.tempdir.name, "gray.png"), gray, 'Gray')
        self._add_image(os.path.join(self.tempdir.name, "thresh.png"), thresh, 'Thresh')
        
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        self.contour = max(contours[0], key=cv.contourArea)

    def _add_image(self, path, img, label):
        
        cv.imwrite(path, img)
        self.images.append((label, path))

    def render(self):

        graph = self.window['graph']
        graph.erase()

        path = None
        for i in self.images:
            if self.window['radio_image_' + i[0]].get():
                path = i[1]

        graph.draw_image(filename=path, location=(0,0))

        if self.contour is not None:
            points = [tuple(xy) for xy in np.squeeze(self.contour)]
            graph.draw_lines(points, color='red', width=2)

    def _init_ui(self):
        
        w, h = self.image_size

        controls = []
        
        for image in self.images:
            label = image[0]
            is_default = 0 == len(controls)
            key = 'radio_image_' + label
            r = sg.Radio(label, 'radio_image', default=is_default, enable_events=True, key=key)
            controls.append(r)
        
        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h-1),
                      graph_top_right = (w-1, 0),
                      background_color='black',
                      key='graph',
                      enable_events=True),
             controls],
        ]
        
        self.window = sg.Window('Perimeter Computer', layout, finalize=True)
        self.render()

    def ui(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event.startswith('radio_image_'):
                self.render()
            else:
                print(event, values)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    pc = PerimeterComputer(args.image, args.output)
    pc.ui()

if __name__ == '__main__':
    main()
