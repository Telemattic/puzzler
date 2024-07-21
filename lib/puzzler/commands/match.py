import math
import numpy as np
import puzzler
import scipy

import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

class Curvature:

    def by_hack(points, stepsize, reverse, ds):

        if reverse:
            points = np.flip(points, axis=0)
        else:
            points = np.vstack((points, points[:len(points)//2]))

        turning_angle = Curvature.compute_turning_angle(points)
            
        cta = np.cumsum(turning_angle)
        h = scipy.signal.firwin(161, 0.005)
        cta = scipy.signal.lfilter(h, 1, cta)

        curvature = np.hstack((np.zeros(ds//2), cta[ds:] - cta[:-ds], np.zeros(ds//2)))
        return Curvature(curvature, points)

    def by_potrace(path, stepsize, reverse=False, ds=None):

        repeat = None if reverse else 0.5
        points = puzzler.potrace.InterpolatePath(stepsize).apply(path, repeat)
        if reverse:
            points = np.flip(points, axis=0)
        
        turning_angle = Curvature.compute_turning_angle(points)
        cum_turning_angle = np.cumsum(turning_angle)

        if ds is not None:
            # assert len(fir_filter) % 2 == 1
            # n = len(fir_filter) // 2
            # fcta = scipy.signal.lfilter(fir_filter, 1., self.cum_turning_angle, axis=0)
            # # print(f"lfilter: len(x)={len(self.cum_turning_angle)} len(y)={len(fcta)}")
            # self.cum_turning_angle = fcta

            cta = cum_turning_angle
            curvature = np.hstack((np.zeros(ds//2), cta[ds:] - cta[:-ds], np.zeros(ds//2)))
        else:
            cum_path_length = np.arange(len(turning_angle)) * stepsize
            curvature = Curvature.compute_curvature(cum_path_length, cum_turning_angle)

        return Curvature(curvature)

    def __init__(self, curvature, points):
        self.curvature = curvature
        self.points = points

    def interp_point(self, cpl):
        # print(f"interp_point: {cpl=} {self.cum_path_length=} {self.points=}")
        x = cpl / self.stepsize
        i = int(x)
        n = len(self.points)
        if i < 0:
            return self.points[0]
        if i >= n-1:
            return self.points[n-1]

        t = x - i
        return (1 - t) * self.points[i] + t * self.points[i+1]

    @staticmethod
    def compute_turning_angle(points):
        
        turning_angle = []
        
        n = len(points)
        for i in range(n):
            p0 = points[i-1] if i > 0 else points[n-1]
            p1 = points[i]
            p2 = points[i+1] if i+1 < n else points[0]
                
            x0, y0 = p2 - p1
            x1, y1 = p1 - p0
            t  = np.arctan2(x0*y1 - y0*x1, x0*x1 + y0*y1)
            turning_angle.append(t)

        return np.array(turning_angle)

    @staticmethod
    def compute_curvature(path_length, cum_turning_angle):

        retval = []
        k = 5
        ds = 50
        smin = path_length.min()
        smax = path_length.max()
        s = smin
        while s < smax:
            samples = np.linspace(s+ds, s+ds+k, num=k)
            c = (np.average(np.interp(samples, path_length, cum_turning_angle)) -
                 np.interp(s, path_length, cum_turning_angle))
            retval.append(c)
            s += 1

        return np.array(retval)

def algorithm_ii(curve_a, curve_b, eps=0.2):
    
    index_a = np.argsort(curve_a)
    
    index_b = np.argsort(curve_b)
    from_ = to = 0
    n = len(index_b)

    count = np.zeros(len(curve_a) + len(curve_b), dtype=np.int32)

    for i in index_a:
        while from_ < n and curve_a[i]-eps > curve_b[index_b[from_]]:
            from_ += 1
        while to < n and curve_a[i]+eps > curve_b[index_b[to]]:
            to += 1

        for j in range(from_,to):
            count[i-index_b[j]+n] += 1

    print(f"max count at i-j={np.argmax(count)-n}, total counts={sum(count)}")

    peaks, _ = scipy.signal.find_peaks(count, distance=150)
    highest = np.max(count[peaks])

    for p in peaks:
        if count[p] > .5 * highest:
            print(f"peak: offset={p-n}, count={count[p]}")
            
    return count

class Camera:

    def __init__(self, center, zoom, viewport):
        self._center = center.copy()
        self._zoom = zoom
        self._viewport = viewport
        self._matrix = None
        self.__update_matrix()

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, v):
        self._center = v.copy()
        self.__update_matrix()

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, v):
        self._zoom = v
        self.__update_matrix()

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, v):
        self._viewport = v
        self.__update_matrix()

    @property
    def matrix(self):
        return self._matrix

    def fixed_point_zoom(self, f, xy):
        xy  = np.array((*xy, 1))
        inv = np.linalg.inv(self._matrix)
        xy  = (xy @ inv.T)[:2]

        self._center = self._zoom * (f - 1) * xy + self._center
        self._zoom   = self._zoom * f
        self.__update_matrix()

    def __update_matrix(self):
        w, h = self.viewport
        
        vp = np.array(((1,  0, w/2),
                       (0, -1, h/2),
                       (0,  0, 1)), dtype=np.float64)

        x, y = self.center
        z = self.zoom
        
        lookat = np.array(((z, 0, -x),
                           (0, z, -y),
                           (0, 0,  1)))

        self._matrix = vp @ lookat

class Draggable:

    def start(self, xy):
        pass

    def drag(self, xy):
        raise NotImplementedError

    def commit(self):
        pass

class MoveCamera(Draggable):

    def __init__(self, camera):
        self.camera = camera
        self.init_camera_center = camera.center.copy()

    def start(self, xy):
        self.origin = xy

    def drag(self, xy):
        delta = (xy - self.origin) * np.array((-1, 1))
        self.camera.center = self.init_camera_center + delta

class CurvesCanvas:

    def __init__(self, parent, var_epsilon, var_offset):

        self.var_epsilon = var_epsilon
        self.var_offset = var_offset

        # https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
        self.figure = Figure(figsize=(8,8), dpi=100, facecolor='white')
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.markers = None
        self.vlines = None

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_notify)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.c1 = None
        self.c2 = None
        self.draggable = None

    def set_curves(self, curves):
        self.c1 = curves[0]
        self.c2 = curves[1]

        self.update()

    def get_dst_points(self):
        if not self.markers:
            return (0, 0)
        return [int(m.get_xdata()[0]) for m in self.markers]

    def get_src_points(self):
        a, b = self.get_dst_points()
        o = self.var_offset.get()
        return [a-o, b-o]

    def update(self):

        epsilon = self.var_epsilon.get()
        offset = self.var_offset.get()
        
        f = self.figure
        for ax in f.axes.copy():
            f.delaxes(ax)
            
        self.ax1 = f.add_subplot(3, 1, 1)
        self.ax2 = f.add_subplot(3, 1, 2)
        self.ax3 = f.add_subplot(3, 1, 3)

        c1 = self.c1.curvature
        c2 = self.c2.curvature
        
        self.ax1.plot(np.arange(1,len(c1)+1), c1)
        self.ax1.plot(np.arange(1,len(c2)+1)+offset, c2)
        self.ax1.set_ylabel('curvature')
        self.ax1.grid(True)

        correlation = algorithm_ii(c1, c2, eps=epsilon)
        lags = np.arange(-len(c2), len(c1))
        
        self.ax2.plot(lags, correlation)
        self.ax2.set_ylabel('correlation')
        self.ax2.set_xlabel('offset')
        self.ax2.grid(True)

        start1, end1 = max(0, offset), min(len(c1), len(c2)+offset)
        start2, end2 = start1-offset, end1-offset

        c3 = np.abs(c1[start1:end1] - c2[start2:end2])
        self.ax3.plot(np.arange(start1,end1), c3, color='tab:green')
        self.ax3.set_ylabel('abs(A - B)')
        self.ax3.grid(True)
        self.markers = [self.ax3.plot([x], [0], marker='o', color='red', picker=True)[0] for x in [start1, end1-1]]
        self.vlines = [self.ax1.axvline(x, ls='--', lw=1, color='red') for x in [start1, end1-1]]

        peaks, _ = scipy.signal.find_peaks(c3, distance=150)
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(c3, peaks)

        self.ax3.plot(peaks+start1, c3[peaks], "x")
        contour_heights = c3[peaks] - prominences
        self.ax3.vlines(x=peaks+start1, ymin=contour_heights, ymax=c3[peaks])

        self.canvas.draw()

    class Draggable:
        def drag(self, e):
            raise NotImplementedError

        def commit(self):
            pass

    class OffsetDraggable(Draggable):
        def __init__(self, e, curves):
            self.inaxes = e.inaxes
            self.xdata0 = e.xdata
            self.offset0 = curves.var_offset.get()
            self.curves = curves

        def drag(self, e):
            if e.inaxes == self.inaxes:
                delta = int(round(e.xdata - self.xdata0))
                self.curves.var_offset.set(self.offset0 + delta)
                self.curves.update_offset()

    class MarkerDraggable(Draggable):
        def __init__(self, e, points, lines):
            self.inaxes = e.mouseevent.inaxes
            self.points = points
            self.lines  = lines

        def drag(self, e):
            if e.inaxes == self.inaxes:
                x = int(round(e.xdata))
                for p in self.points:
                    p.set_xdata([x])
                for l in self.lines:
                    l.set_xdata([x])

    def on_button_press(self, e):
        if e.inaxes == self.ax1:
            self.draggable = CurvesCanvas.OffsetDraggable(e, self)

    def on_button_release(self, e):
        if self.draggable:
            self.draggable.commit()
            self.draggable = None
            self.canvas.draw_idle()

    def on_motion_notify(self, e):
        if self.draggable:
            self.draggable.drag(e)
            self.canvas.draw_idle()

    def on_pick(self, e):
        m = e.mouseevent
        if e.artist and m.inaxes == self.ax3:
            points = [e.artist]
            if e.artist == self.markers[0]:
                lines = [self.vlines[0]]
            else:
                lines = [self.vlines[1]]
            self.draggable = CurvesCanvas.MarkerDraggable(e, points, lines)
        
    def update_offset(self):
        offset = self.var_offset.get()

        c1 = self.c1.curvature
        c2 = self.c2.curvature
        start1, end1 = max(0, offset), min(len(c1), len(c2)+offset)
        start2, end2 = start1-offset, end1-offset

        self.ax1.lines[1].set_xdata(np.arange(1,len(c2)+1)+offset)

        c3 = np.abs(c1[start1:end1] - c2[start2:end2])
        self.ax3.lines[0].set_data(np.arange(start1, end1), c3)
        
        self.canvas.draw_idle()

class MatchTk:

    def __init__(self, parent, puzzle, labels):

        pieces = dict()
        coords = dict()
        for p in puzzle.pieces:
            if p.label in labels:
                x = 1000. * len(pieces)
                y = 0.
                pieces[p.label] = p
                coords[p.label] = puzzler.align.Coord(0., (x,y))
        
        self.puzzle = puzzle
        self.labels = labels
        self.pieces = pieces
        self.coords = coords
        self.curves = dict()

        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.puzzle_canvas = tkinter.Canvas(self.frame, width=800, height=800,
                             background='white', highlightthickness=0)
        self.puzzle_canvas.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.S))
        self.puzzle_canvas.bind("<Button-1>", self.on_puzzle_press)
        self.puzzle_canvas.bind("<B1-Motion>", self.on_puzzle_drag)
        self.puzzle_canvas.bind("<ButtonRelease-1>", self.on_puzzle_release)
        self.puzzle_canvas.bind("<MouseWheel>", self.on_puzzle_wheel)

        self.var_epsilon = tkinter.DoubleVar(value=0.2)
        self.var_offset = tkinter.IntVar(value=0)
        
        self.curves_canvas = CurvesCanvas(parent, self.var_epsilon, self.var_offset)
        self.curves_canvas.canvas.get_tk_widget().grid(
            column=1, row=0, sticky=(tkinter.N, tkinter.E, tkinter.S, tkinter.W))
       
        self.controls = ttk.Frame(self.frame)
        self.controls.grid(column=0, row=1, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S), pady=5)

        ttk.Button(self.controls, text='potrace', command=self.potrace).grid(column=0, row=0)

        ttk.Label(self.controls, text='Stepsize').grid(column=2, row=0)

        self.var_stepsize = tkinter.DoubleVar(value=1.)
        ttk.Entry(self.controls, width=8, textvariable=self.var_stepsize).grid(column=3, row=0)

        self.var_render_potrace_path = tkinter.IntVar(value=0)
        ttk.Checkbutton(self.controls, text='Path', command=self.render,
                        variable=self.var_render_potrace_path).grid(column=4, row=0)

        self.var_render_potrace_points = tkinter.IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Points', command=self.render,
                        variable=self.var_render_potrace_points).grid(column=5, row=0)

        self.var_render_potrace_lines = tkinter.IntVar(value=0)
        ttk.Checkbutton(self.controls, text='Lines', command=self.render,
                        variable=self.var_render_potrace_lines).grid(column=6, row=0)

        self.var_render_perimeter = tkinter.IntVar(value=1)
        ttk.Checkbutton(self.controls, text='Perimeter', command=self.render,
                        variable=self.var_render_perimeter).grid(column=7, row=0)

        self.var_ds = tkinter.IntVar(value=50)
        ttk.Label(self.controls, text='ds').grid(column=8, row=0)
        ttk.Entry(self.controls, width=5, textvariable=self.var_ds).grid(column=9, row=0)

        ttk.Label(self.controls, text='epsilon').grid(column=10, row=0)
        ttk.Entry(self.controls, width=5, textvariable=self.var_epsilon).grid(column=11, row=0)

        ttk.Label(self.controls, text='offset').grid(column=12, row=0)
        ttk.Entry(self.controls, width=8, textvariable=self.var_offset).grid(column=13, row=0)

        ttk.Button(self.controls, text='align', command=self.align).grid(column=0, row=1)

        self.potrace_path = None
        self.cursor_xdata = None
        self.draggable = None
        self.camera = Camera(np.array((0,0), dtype=np.float64), 1/3, (800,800))
        self.render()

    def align(self):
        print("Align!")

        dst_curve = self.curves[self.labels[0]]
        src_curve = self.curves[self.labels[1]]

        dst_piece = self.pieces[self.labels[0]]
        src_piece = self.pieces[self.labels[1]]

        dst_points = dst_curve.points[self.curves_canvas.get_dst_points()]
        src_points = src_curve.points[self.curves_canvas.get_src_points()]

        dst_vec = dst_points[1] - dst_points[0]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[1] - src_points[0]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = puzzler.align.Coord(dst_angle-src_angle).xform.apply_v2(src_points)
        r, x, y = puzzler.align.compute_rigid_transform(src_points_rotated, dst_points)
        r += dst_angle - src_angle
        print(f"rigid_transform: {r=:.3f} {x=:.1f} {y=:.1f}")

        self.coords[self.labels[1]] = puzzler.align.Coord(r, (x,y))
        self.render()

    def on_puzzle_press(self, event):
        self.draggable = MoveCamera(self.camera)
        self.draggable.start(np.array((event.x, event.y)))
        self.render()

    def on_puzzle_drag(self, event):
        if self.draggable:
            self.draggable.drag(np.array((event.x, event.y)))
            self.render()

    def on_puzzle_release(self, event):
        if self.draggable:
            self.draggable.commit()
            self.draggable = None
            self.render()

    def on_puzzle_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.render()

    def on_figure_motion(self, event):
        if event.inaxes:
            self.cursor_xdata = event.xdata
            self.update_cursor()
        
    def potrace(self):

        stepsize = self.var_stepsize.get()
        ds = self.var_ds.get()
        
        for label in self.labels:
            piece = self.pieces[label]
            rev = label != self.labels[0]
            if True:
                self.curves[label] = Curvature.by_hack(piece.points, stepsize, rev, ds)
            else:
                path = puzzler.potrace.piece_to_path(piece)[0]
                self.curves[label] = Curvature.by_potrace(path, stepsize, reverse=rev, ds=ds)
            
        self.render()

        c1 = self.curves[self.labels[0]]
        c2 = self.curves[self.labels[1]]

        self.curves_canvas.set_curves([c1,c2])

    def render(self):
        canvas = self.puzzle_canvas
        canvas.delete('all')

        colors = list(matplotlib.colors.TABLEAU_COLORS.values())

        r = puzzler.renderer.canvas.CanvasRenderer(canvas)

        r.transform(self.camera.matrix)

        for i, label in enumerate(self.labels):

            piece = self.pieces[label]
            coord = self.coords[label]
            
            with puzzler.render.save(r):
            
                r.translate(coord.xy)
                r.rotate(coord.angle)

                outline = colors[i % len(colors)]
                r.draw_polygon(piece.points, outline=outline, fill='', width=2)

                r.draw_text(np.array((0,0)), label)

        self.update_cursor()

    def update_cursor(self):

        canvas = self.puzzle_canvas
        canvas.delete('cursor')
        if self.cursor_xdata is None:
            return

        for l in self.labels:
            o = 0 if l == self.labels[0] else self.var_offset.get()
            c = self.curves[l].interp_point(self.cursor_xdata - o)
            r = puzzler.renderer.canvas.CanvasRenderer(canvas)
            r.transform(self.camera.matrix)
            coord = self.coords[l]
            r.translate(coord.xy)
            r.rotate(coord.angle)
            r.draw_circle(c, radius=6, fill='', outline='red', tag='cursor')
    
def match(args):

    puzzle = puzzler.file.load(args.puzzle)
    root = tkinter.Tk()
    ui = MatchTk(root, puzzle, args.labels)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: match")
    root.mainloop()
    
def add_parser(commands):
    parser_match = commands.add_parser("match", help="curve matching")
    parser_match.add_argument("labels", nargs=2, help="two pieces to match to each other")
    parser_match.set_defaults(func=match)
