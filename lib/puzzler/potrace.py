import cv2 as cv
from dataclasses import dataclass
import numpy as np
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

def format_point(p):
    return f"({p[0]:.1f},{p[1]:.1f})"

@dataclass
class Line:
    v0: "tuple[float,float]"
    v1: "tuple[float,float]"

    def __repr__(self):
        v0 = format_point(self.v0)
        v1 = format_point(self.v1)
        return f"Line({v0},{v1})"

@dataclass
class Spline:
    p0: "tuple[float,float]"
    p1: "tuple[float,float]"
    p2: "tuple[float,float]"
    p3: "tuple[float,float]"

    def __repr__(self):
        p0 = format_point(self.p0)
        p1 = format_point(self.p1)
        p2 = format_point(self.p2)
        p3 = format_point(self.p3)
        return f"Spline({p0},{p1},{p2},{p3})"

class Tokens:

    def __init__(self, path):
        self.data = [i for i in re.split(r'([a-zA-Z\s])', path) if i != '' and not i.isspace()]
        self.i = 0

    def is_end(self):
        return self.i >= len(self.data)

    def is_command(self):
        return self.data[self.i].isalpha()

    def get_command(self):
        self.i += 1
        return self.data[self.i-1]

    def get_int(self):
        self.i += 1
        return int(self.data[self.i-1])

def parse_path(path, ll):

    retval = []
    tokens = Tokens(path)

    def P(x, y):
        return (0.1 * x + ll[0], -0.1 * y + ll[1])

    command = None
    cx, cy = 0, 0
    subpath = []
    while not tokens.is_end():

        if command is None or tokens.is_command():
            command = tokens.get_command()

        if command == 'M':
            cx = tokens.get_int()
            cy = tokens.get_int()
        elif command == 'm':
            cx += tokens.get_int()
            cy += tokens.get_int()
        elif command == 'l':
            x = cx + tokens.get_int()
            y = cy + tokens.get_int()
            subpath.append(Line(P(cx, cy), P(x,y)))
            cx, cy = x, y
        elif command == 'c':
            x1 = cx + tokens.get_int()
            y1 = cy + tokens.get_int()
            x2 = cx + tokens.get_int()
            y2 = cy + tokens.get_int()
            x = cx + tokens.get_int()
            y = cy + tokens.get_int()
            subpath.append(Spline(P(cx, cy), P(x1,y1), P(x2,y2), P(x,y)))
            cx, cy = x, y
        elif command == 'z':
            retval.append(subpath)
            subpath = []
        else:
            raise ValueError("unknown svg path command")

    if len(subpath):
        retval.append(subpath)
        subpath = []
            
    return retval

def piece_to_input_image(piece, fpath, pad=2):

    (x1, y1), (x2, y2) = piece.bbox
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    points = piece.points - (x1,y1)

    w, h = x2 - x1, y2 - y1
    
    image = np.zeros((h, w), dtype=np.uint8)

    image[points[:,1],points[:,0]] = 255
    cv.floodFill(image, None, (0,0), 255)

    cv.imwrite(fpath, image)

    return (x1, y2)

def piece_to_path(piece, tempdir=None):

    if tempdir is None:
        tempdir = tempfile.TemporaryDirectory(dir='C:\\Temp', prefix='potrace')
        
    ipath = os.path.join(tempdir.name, 'potrace_input.pgm')
    
    ll = piece_to_input_image(piece, ipath)

    opath = os.path.join(tempdir.name, 'potrace_output.svg')

    return image_to_path(ipath, opath, ll)

def image_to_path(ipath, opath, ll=(0,0), args=None):

    exe = r'C:\temp\potrace-1.16.win64\potrace.exe'
    
    allargs = [exe]
    
    if args is not None:
        allargs += args
        
    allargs += ['-b' 'svg', '-o', opath, '--', ipath]
    
    subprocess.run(allargs)

    tree = ET.parse(opath)
    root = tree.getroot()

    ns = {'svg':'http://www.w3.org/2000/svg'}

    svgpath = []
    for i in root.find('svg:g', ns).findall('svg:path', ns):
        svgpath += parse_path(i.attrib['d'], ll)
    
    return svgpath

class InterpolatePath:

    # Great resource for bezier curves:
    #
    # https://pomax.github.io/bezierinfo/#derivatives

    def __init__(self, stepsize):
        # how big are the steps
        self.stepsize = stepsize
        # arc length distance to the next step
        self.nextstep = 0.
        self.retval = []

    def apply(self, path, repeat=None):

        for i in path:
            if isinstance(i,Line):
                self.interpolate_line(i)
            else:
                self.interpolate_spline(i)

        if repeat is not None:
            n = int(len(self.retval) * (1 + repeat))
            for i in path:
                if isinstance(i,Line):
                    self.interpolate_line(i)
                else:
                    self.interpolate_spline(i)
                if len(self.retval) >= n:
                    break
    
        return np.array(self.retval)

    def interpolate_line(self, line):
        
        v0 = np.array(line.v0)
        v1 = np.array(line.v1)
        ll = np.linalg.norm(v1 - v0)
        uv = (v1 - v0) / ll

        i = 0
        while ll > self.nextstep:
            v0 = v0 + uv * self.nextstep
            ll -= self.nextstep
            self.retval.append(v0)
            i += 1
            self.nextstep = self.stepsize

        self.nextstep -= ll

    def interpolate_spline(self, spline):

        control_points = np.array([spline.p0, spline.p1, spline.p2, spline.p3])
        v0 = np.array(spline.p0)
        n = 50
        for i in range(n):
            t = (i+1)/n 
            v1 = self.spline_eval(control_points, t)
            ll = np.linalg.norm(v1 - v0)
            if ll == 0.:
                v0 = v1
                continue
            
            uv = (v1 - v0) / ll
            while ll > self.nextstep:
                v0 = v0 + uv * self.nextstep
                ll -= self.nextstep
                self.retval.append(v0)
                self.nextstep = self.stepsize
                    
            self.nextstep -= ll
            v0 = v1

    @staticmethod
    def spline_eval(control_points, t):
        while len(control_points) > 1:
            p1 = control_points[:-1]
            p2 = control_points[1:]
            control_points = (1-t)*p1 + t*p2
        return control_points[0]
