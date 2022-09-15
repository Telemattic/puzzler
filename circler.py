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
        
        self.rot90 = img.shape[0] > img.shape[1]
        if self.rot90:
            img = np.rot90(img)
        
        w, h = img.shape[1], img.shape[0]
        self.image_size = (w,h)

        print(f"image: {self.image_path}, {w} x {h}")

        self.images.add('color.png', img, 'Source')
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        self.images.add('gray.png', gray, 'Gray')

        thresh = cv.threshold(gray, 84, 255, cv.THRESH_BINARY)[1]
        
        self.images.add('thresh.png', thresh, 'Thresh')

        morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel=np.ones((9,9), np.uint8))

        self.images.add('morph.png', morph, 'Morph')

        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        assert 2 == len(contours)
        
        contour = np.squeeze(max(contours[0], key=cv.contourArea))

        self.perimeter_points = contour.copy()
        self.perimeter_index  = dict((tuple(xy), i) for i, xy in enumerate(self.perimeter_points))

        self.tree0 = scipy.spatial.KDTree(self.perimeter_points)

        rows, cols = contour[:,1], contour[:,0]
        
        contour_img = np.full((h, w), 255, dtype=np.uint8)
        contour_img[rows, cols] = 0

        self.images.add('contour0.png', contour_img, 'Contour 0')
        
        distance_f = cv.distanceTransform(contour_img, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        contour_img = np.full((h, w, 3), 255, dtype=np.uint8)
        contour_img[rows, cols] = np.uint8([0, 0, 255])

        self.counts_by_threshold = []

        for t in [5, 10, 15, 20, 30, 40, 50]:
            
            thresh = np.uint8(cv.threshold(distance_f, t, 255, cv.THRESH_BINARY_INV)[1])
            contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

            print(f"{t=} {len(contours)=} {hierarchy.tolist()=}")
            
            counts_by_contour = []
            
            for i, c in enumerate(contours):
                print(f"{t=} {i=} {cv.contourArea(c)=:.1f}")
                c = np.squeeze(c)
                rows, cols = c[:,1], c[:,0]
                contour_img[rows, cols] = np.uint8([0,0,0])

                treeN = scipy.spatial.KDTree(c)
                indexes = self.tree0.query_ball_tree(treeN, r=t*1.1)

                counts_by_contour.append([len(i) for i in indexes])

            self.counts_by_threshold.append((t, counts_by_contour))

        self.images.add('contourN.png', contour_img, 'Contour N')

        distance_u8 = np.uint8(distance_f * (255. / distance_f.max()))

        self.images.add('distance.png', distance_u8, 'Distance')

        contour_img2 = np.full((h, w), 255, dtype=np.uint8)
        contour_img2[rows, cols] = 0

        distance_f2 = cv.distanceTransform(contour_img2, cv.DIST_L2, cv.DIST_MASK_PRECISE)
        thresh2 = np.uint8(cv.threshold(distance_f2, t, 255, cv.THRESH_BINARY_INV)[1])
        
        contours, hierarchy = cv.findContours(thresh2, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        # recomputed outer contour
        c = np.squeeze(contours[0])
        rows, cols = c[:,1], c[:,0]
        contour_img[rows, cols] = np.uint8([0, 180, 0])

        self.images.add('contourQ.png', contour_img, 'Contour Q')

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

    def draw_trace(self):
        
        trace = self.window['trace']

        max_count = max(max(max(count) for count in counts) for t, counts in self.counts_by_threshold)
        print(f"{max_count=}")
        for t, counts in self.counts_by_threshold[-1:]:
            h = 100. / len(counts)
            for i, count in enumerate(counts):
                y = 100. - (i+1)*h
                s = h / max_count
                points = [(i, y + v * s) for i, v in enumerate(count)]
                trace.draw_lines(points, color='black')

            diff = [a - b for a, b in zip(counts[0], counts[1])]
            max_diff = max(diff)
            min_diff = min(diff)
            print(f"{t=} {max_diff=} {min_diff=}")
            s = 50 / max(max_diff, -min_diff)
            points = [(i, 50 + v * s) for i, v in enumerate(diff)]
            trace.draw_lines(points, color='yellow')

    def draw_trace2(self):

        trace = self.window['trace']
        
        points = self.perimeter_points
        center = np.mean(points, axis=0)
        dist   = np.hypot(points[:,0]-center[0], points[:,1]-center[1])
        maxdist = np.max(dist)
        plot_points = [(i,v) for i, v in enumerate(dist * (100 / maxdist))]
        trace.draw_lines(plot_points, color='purple')
        
    def render(self):

        graph = self.window['graph']
        graph.erase()
        self.hover_text = None
        self.trace_marker = None

        path = self.get_image_path()
        graph.draw_image(filename=path, location=(0,0))

        self.draw_trace2()

        if self.window['render_histogram'].get():
            self.draw_histogram()

    def clear_hover_text(self):
        
        if self.hover_text is not None:
            graph = self.window['graph']
            graph.delete_figure(self.hover_text)
            self.hover_text = None
            
    def update_hover_text(self, i):

        x, y = self.perimeter_points[i].tolist()
        text = f"{i=}: {x},{y}"
        for t, counts in self.counts_by_threshold:
            counts_text = ', '.join(f"{v[i]}" for v in counts)
            text += f"\n  {t=}: counts={counts_text}"

        graph = self.window['graph']
        
        self.clear_hover_text()

        if False:
            self.hover_text = graph.draw_text(text, (x,y), font=('Courier', 12, 'bold'), color='purple', text_location=sg.TEXT_LOCATION_BOTTOM_LEFT)
        else:
            self.hover_text = graph.draw_circle((x,y), 10, line_color='purple')

    def clear_trace_marker(self):

        if self.trace_marker is not None:
            trace = self.window['trace']
            trace.delete_figure(self.trace_marker)
            self.trace_marker = None

    def update_trace_marker(self, i):

        trace = self.window['trace']
        self.clear_trace_marker()
        self.trace_marker = trace.draw_line((i, 0), (i,100), width=2, color='purple')

    def graph_motion_callback(self, e):

        d, i = self.tree0.query([e.x, e.y])
        if d > 5:
            self.clear_hover_text()
            self.clear_trace_marker()
            return

        self.update_hover_text(i)
        self.update_trace_marker(i)

    def graph_leave_callback(self, e):
        self.clear_hover_text()
        self.clear_trace_marker()

    def trace_motion_callback(self, e):

        i, _ = self.window['trace']._convert_canvas_xy_to_xy(e.x, e.y)

        if 0 <= i < self.perimeter_points.shape[0]:
            self.update_hover_text(i)
            self.update_trace_marker(i)
        else:
            self.clear_hover_text()
            self.clear_trace_marker()

    def trace_leave_callback(self, e):
        self.clear_hover_text()
        self.clear_trace_marker()
        
    def _init_ui(self):
        
        w, h = self.image_size

        controls = []
        
        for label, path in self.images.images.items():
            is_default = 0 == len(controls)
            key = 'radio_image_' + label
            r = sg.Radio(label, 'radio_image', default=is_default, enable_events=True, key=key)
            controls.append(r)

        controls.append(sg.CB('Render Histogram', default=False, enable_events=True, key='render_histogram'))

        graph = sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, h-1),
                      graph_top_right = (w-1, 0),
                      background_color='black',
                      key='graph',
                      enable_events=True)

        w, h = self.perimeter_points.shape[0], 100

        trace = sg.Graph(canvas_size = (self.image_size[0], 200),
                         graph_bottom_left = (0,0),
                         graph_top_right = (w, h),
                         background_color = 'white',
                         key='trace',
                         enable_events=True)

        layout = [
            [controls],
            [trace],
            [graph]
        ]
        
        self.window = sg.Window('Perimeter Computer', layout, finalize=True)

        graph._TKCanvas2.bind('<Motion>', self.graph_motion_callback)
        graph._TKCanvas2.bind('<Leave>', self.graph_leave_callback)
        
        trace._TKCanvas2.bind('<Motion>', self.trace_motion_callback)
        trace._TKCanvas2.bind('<Leave>', self.trace_leave_callback)

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
