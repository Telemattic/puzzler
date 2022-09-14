import argparse
import cv2 as cv
import json
import math
import numpy as np
import os
import scipy
import PySimpleGUI as sg
import tempfile

class TempImages:

    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp')
        self.images  = dict()

    def add(self, filename, img, label):
        
        path = os.path.join(self.tempdir.name, filename)
        ret  = cv.imwrite(path, img)
        assert ret is not None
        
        self.images[label] = path
        
class PerimeterComputer:

    def __init__(self, image_path, output_path = None):
        
        self.image_path = image_path
        self.images     = TempImages()

        img = cv.imread(self.image_path)
        assert img is not None
        
        self.transposed = img.shape[0] > img.shape[1]
        if self.transposed:
            before = img.shape
            img = np.swapaxes(img, 0, 1)
            print(f"transpose: {before=} after={img.shape}")
        
        w, h = img.shape[1], img.shape[0]
        self.image_size  = (w,h)

        print(f"image: {self.image_path}, {w} x {h}")

        self.images.add('color.png', img, 'Source')
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        self.images.add('gray.png', gray, 'Gray')

        thresh = cv.threshold(gray, 84, 255, cv.THRESH_BINARY)[1]
        
        self.images.add('thresh.png', thresh, 'Thresh')

        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        assert 2 == len(contours)
        
        contour = max(contours[0], key=cv.contourArea)

        rows, cols = np.squeeze(contour)[:,1], np.squeeze(contour)[:,0]
        
        contour_img = np.full((h, w), 255, dtype=np.uint8)
        contour_img[rows, cols] = 0

        self.images.add('contour0.png', contour_img, 'Contour 0')
        
        distance_f = cv.distanceTransform(contour_img, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        contour_img = np.full((h, w, 3), 255, dtype=np.uint8)
        contour_img[rows, cols] = np.uint8([0, 0, 255])

        for t in [5, 10, 15, 20, 30, 40, 50]:
            
            thresh = np.uint8(cv.threshold(distance_f, t, 255, cv.THRESH_BINARY_INV)[1])
            contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
            for c in contours:
                rows, cols = np.squeeze(c)[:,1], np.squeeze(c)[:,0]
                contour_img[rows, cols] = np.uint8([0,0,0])

        self.images.add('contourN.png', contour_img, 'Contour N')

        distance_u8 = np.uint8(distance_f * (255. / distance_f.max()))

        self.images.add('distance.png', distance_u8, 'Distance')

    def get_image_path(self):

        for label, path in self.images.images.items():
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

    def draw_approx(self):
        graph = self.window['graph']
        
        graph.draw_lines(self.approx, color='yellow', width=2)

        for i, xy in enumerate(self.approx):
            graph.draw_point(xy, size=5, color='purple')

        for i, xy in enumerate(self.approx):
            x, y = xy
            graph.draw_text(f"{i}", (x+10,y+10), color='green')

    def render(self):

        graph = self.window['graph']
        graph.erase()
        self.hover_text = None

        path = self.get_image_path()
        graph.draw_image(filename=path, location=(0,0))

        if self.window['render_traces'].get():
            self.draw_traces()

        if self.window['render_histogram'].get():
            self.draw_histogram()

    def update_hover_text(self, text, xy):

        graph = self.window['graph']
        
        if self.hover_text is not None:
            graph.delete_figure(self.hover_text)
            self.hover_text = None

        self.hover_text = graph.draw_text(text, xy, font=('Courier', 12, 'bold'), color='purple', text_location=sg.TEXT_LOCATION_BOTTOM_LEFT)

    def motion_callback(self, e):

        xy = (e.x, e.y)
        self.update_hover_text(f"{xy}", xy)

    def leave_callback(self, e):

        graph = self.window['graph']
        
        if self.hover_text is not None:
            graph.delete_figure(self.hover_text)
            self.hover_text = None

    def _init_ui(self):
        
        w, h = self.image_size

        controls = []
        
        for label, path in self.images.images.items():
            is_default = 0 == len(controls)
            key = 'radio_image_' + label
            r = sg.Radio(label, 'radio_image', default=is_default, enable_events=True, key=key)
            controls.append(r)

        controls.append(sg.CB('Render Histogram', default=False, enable_events=True, key='render_histogram'))
        controls.append(sg.CB('Render Traces', default=False, enable_events=True, key='render_traces'))

        graph = sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h-1),
                      graph_top_right = (w-1, 0),
                      background_color='black',
                      key='graph',
                      enable_events=True)

        layout = [
            [graph,
             controls],
        ]
        
        self.window = sg.Window('Perimeter Computer', layout, finalize=True)

        graph._TKCanvas2.bind('<Motion>', self.motion_callback)
        graph._TKCanvas2.bind('<Leave>', self.leave_callback)
        
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
