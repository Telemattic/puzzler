import argparse
import cv2 as cv
import json
import numpy as np
import os
import PySimpleGUI as sg
import tempfile

import scipy
import scipy.ndimage as ndimage

def get_corners(dst, neighborhood_size=5, score_threshold=0.3, minmax_threshold=100):
    
    """
    Given the input Harris image (where in each pixel the Harris function is computed),
    extract discrete corners
    """
    data = dst.copy()
    data[data < score_threshold*dst.max()] = 0.

    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > minmax_threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
    return yx

class PerimeterComputer:

    def __init__(self, image_path, output_path = None):
        
        self.image_path  = image_path
        self.output_path = output_path
        
        self.tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp')

        img = cv.imread(self.image_path)
        assert img is not None
        
        w, h = img.shape[1], img.shape[0]
        self.image_size  = (w,h)

        print(f"image={self.image_path} {w}x{h}")
        
        img      = cv.resize(img, self.image_size, cv.INTER_CUBIC)
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray     = cv.medianBlur(gray, 7)
        gray     = cv.GaussianBlur(gray, (3,3), 0)
        thresh   = 255 - cv.threshold(gray, 84, 255, cv.THRESH_BINARY_INV)[1]
        thresh   = cv.medianBlur(thresh, 29)

        harris   = cv.dilate(cv.cornerHarris(np.float32(thresh), 5, 5, .004), None)
        corners  = np.zeros(img.shape, dtype=np.uint8)
        corners[harris > .1*harris.max()] = [0,255,0]

        print(f"{harris.min()=} {harris.max()=}")

        xycorners = get_corners(harris)
        print(f"{xycorners=}")

        harris = np.uint8((harris - harris.min()) * 255 / (harris.max() - harris.min()))
        print(f"{harris=}")

        print(f"temp={self.tempdir.name}")

        self.images = []
        self._add_image(os.path.join(self.tempdir.name, "color.png"), img, 'Source')
        self._add_image(os.path.join(self.tempdir.name, "gray.png"), gray, 'Gray')
        self._add_image(os.path.join(self.tempdir.name, "thresh.png"), thresh, 'Thresh')
        self._add_image(os.path.join(self.tempdir.name, "harris.png"), harris, 'Harris')
        self._add_image(os.path.join(self.tempdir.name, "corners.png"), corners, 'Corners')
        
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        self.contour = max(contours[0], key=cv.contourArea)

        if self.output_path:
            data = {
                'bbox': [0, 0, w, h],
                'contour': np.squeeze(self.contour).tolist()
            }
            with open(self.output_path, 'w') as f:
                json.dump(data, f)

    def _add_image(self, path, img, label):
        
        cv.imwrite(path, img)
        self.images.append((label, path))

    def get_image_path(self):

        for label, path in self.images:
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

        controls.append(sg.CB('Render Histogram', default=False, enable_events=True, key='render_histogram'))
        controls.append(sg.CB('Render Traces', default=False, enable_events=True, key='render_traces'))
        controls.append(sg.CB('Render Contour', default=True, enable_events=True, key='render_contour'))
        
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
            elif event.startswith('render_'):
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
