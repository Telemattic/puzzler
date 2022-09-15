import argparse
import cv2 as cv
import json
import math
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
        self.images = []

        print(f"image={self.image_path} {w}x{h}")
        
        img      = cv.resize(img, self.image_size, cv.INTER_CUBIC)
        gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        self.distanceMap(gray)

        gray     = cv.medianBlur(gray, 7)
        gray     = cv.GaussianBlur(gray, (3,3), 0)
        
        self.find_sharp_corners(gray, 1)
        self.approximate_curvature()

        thresh   = 255 - cv.threshold(gray, 60, 255, cv.THRESH_BINARY_INV)[1]
        # thresh   = cv.medianBlur(thresh, 29)

        harris   = cv.dilate(cv.cornerHarris(np.float32(thresh), 5, 5, .004), None)
        corners  = np.zeros(img.shape, dtype=np.uint8)
        corners[harris > .1*harris.max()] = [0,255,0]

        print(f"{harris.min()=} {harris.max()=}")

        xycorners = get_corners(harris)
        print(f"{xycorners=}")

        harris = np.uint8((harris - harris.min()) * 255 / (harris.max() - harris.min()))
        print(f"{harris=}")

        print(f"temp={self.tempdir.name}")

        self._add_temp_image("color.png", img, 'Source')
        self._add_temp_image("gray.png", gray, 'Gray')
        self._add_temp_image("thresh.png", thresh, 'Thresh')
        self._add_temp_image("harris.png", harris, 'Harris')
        self._add_temp_image("corners.png", corners, 'Corners')
        
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        self.contour = max(contours[0], key=cv.contourArea)

        if self.output_path:
            self._save_json(self.output_path)

    @staticmethod
    def compute_support(prev, curr, succ):

        x0, y0 = curr
        x1, y1 = prev
        x2, y2 = succ
        # distance(P1,P2,(x0,y0))

        l = math.hypot(x2-x1, y2-y1)
        d = ((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / l
        return (d, l)

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def approximate_curvature(self):

        points = self.approx
        n = len(points)
        curvature = []
        for i, xy in enumerate(points):

            p, s = 1, 1
            d, l = self.compute_support(points[i-p], xy, points[(i+s)%n])
            while True:

                p1, s1 = p, s
                
                if self.distance(xy, points[i-p1]) < self.distance(xy, points[(i+s1)%n]):
                    p1 += 1
                else:
                    s1 += 1
                
                d1, l1 = self.compute_support(points[i-p1], xy, points[(i+s1)%n])

                # print(f" ... {i=}: {p=} {s=} {d=:.3f} {l=:.3f} {p1=} {s1=} {d1=:.3f} {l1=:.3f}")
                
                if l > l1:
                    break

                if d > 0 and d/l >= d1/l1:
                    break

                if d < 0 and d/l <= d1/l1:
                    break

                p, s, d, l = p1, s1, d1, l1

            curvature.append((p, s, d, l))

        self.curvature = curvature
        
        for i, c in enumerate(curvature):
            p, s, d, l = c
            if p+s > 3:
                print(f"curv[{i}]: {p=} {s=} {d=:.3f} {l=:.3f}")

    def find_sharp_corners(self, gray, epsilon):

        scale = 1
        if scale != 1:
            print(f"{gray.shape=}")
            gray = cv.resize(gray, dsize=(gray.shape[1]//scale,gray.shape[0]//scale),  interpolation=cv.INTER_AREA)
            print(f"{gray.shape=}")

        thresh   = 255 - cv.threshold(gray, 84, 255, cv.THRESH_BINARY_INV)[1]
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        assert isinstance(contours, tuple) and 2 == len(contours)

        contour = max(contours[0], key=cv.contourArea)
        print(f"contour: {contour.shape[0]} vertexes")
        
        approx  = cv.approxPolyDP(contour, epsilon, True)

        self.approx = [tuple(xy) for xy in np.squeeze(approx)*scale]

        print(f"approxPoly: {len(self.approx)} vertexes")

    def distanceMap(self, gray):

        thresh   = 255 - cv.threshold(gray, 84, 255, cv.THRESH_BINARY_INV)[1]
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        contour = max(contours[0], key=cv.contourArea)
        print(f"contour: {contour.shape[0]} vertexes")

        print(f"{contour=}")

        contour_image = np.zeros(thresh.shape, np.uint8) + 255
        # np.put(contour_image, contour, 0)
        for x, y in np.squeeze(contour):
            contour_image[y, x] = 0

        self._add_temp_image("contour_image.png", contour_image, "Contour Image")

        distance_f = cv.distanceTransform(contour_image, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        for t in [5, 10, 15, 20, 30, 40, 50]:
            thresh = np.uint8(cv.threshold(distance_f, t, 255, cv.THRESH_BINARY_INV)[1])
            # print(f"{thresh=}")
            
            contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
            for c in contours:
                for x,y in np.squeeze(c):
                    contour_image[y,x] = 0

        self._add_temp_image("contour_t.png", contour_image, "Contour T")

        distance_u8 = np.uint8(distance_f * (255. / distance_f.max()))
        
        self._add_temp_image("distance.png", distance_u8, "Distance")
        
    def _save_json(self, path):

        data = {
            'bbox': [0, 0, w, h],
            'contour': np.squeeze(self.contour).tolist()
        }
        with open(self.output_path, 'w') as f:
            json.dump(data, f)

    def _add_temp_image(self, filename, img, label):

        path = os.path.join(self.tempdir.name, filename)
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

    def draw_approx(self):
        graph = self.window['graph']
        
        graph.draw_lines(self.approx, color='yellow', width=2)

        for i, xy in enumerate(self.approx):
            graph.draw_point(xy, size=5, color='purple')

        for i, xy in enumerate(self.approx):
            x, y = xy
            graph.draw_text(f"{i}", (x+10,y+10), color='green')

    def draw_contour(self):
        graph = self.window['graph']
        
        points = [tuple(xy) for xy in np.squeeze(self.contour)]
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

        if self.window['render_approx'].get():
            self.draw_approx()

        if self.window['render_contour'].get():
            self.draw_contour()

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
        controls.append(sg.CB('Render Approx', default=False, enable_events=True, key='render_approx'))
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
