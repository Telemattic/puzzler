import numpy as np
import puzzler
import scipy

import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

class Curvature:

    def __init__(self, path, stepsize, reverse=False, ds=None):

        self.points = puzzler.potrace.InterpolatePath(stepsize).apply(path)
        if reverse:
            self.points = np.flip(self.points, axis=0)
        
        self.stepsize = stepsize
        
        self.turning_angle = self.compute_turning_angle(self.points)
        self.cum_turning_angle = np.cumsum(self.turning_angle)
        self.cum_path_length = np.arange(len(self.turning_angle)) * self.stepsize

        if ds is not None:
            # assert len(fir_filter) % 2 == 1
            # n = len(fir_filter) // 2
            # fcta = scipy.signal.lfilter(fir_filter, 1., self.cum_turning_angle, axis=0)
            # # print(f"lfilter: len(x)={len(self.cum_turning_angle)} len(y)={len(fcta)}")
            # self.cum_turning_angle = fcta

            cta = self.cum_turning_angle
            self.curvature = np.hstack((np.zeros(ds//2), cta[ds:] - cta[:-ds], np.zeros(ds//2)))
        else:
            self.curvature = self.compute_curvature(self.cum_path_length, self.cum_turning_angle)

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

class MatchTk:

    def __init__(self, parent, puzzle, labels):

        piece = None
        for p in puzzle.pieces:
            if p.label == labels[0]:
                piece = p
        assert piece is not None
        
        self.puzzle = puzzle
        self.labels = labels
        self.piece = piece

        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.puzzle_canvas = tkinter.Canvas(self.frame, width=800, height=800,
                             background='white', highlightthickness=0)
        self.puzzle_canvas.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.S))

        # https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
        self.figure = Figure(figsize=(8,8), dpi=100, facecolor='white')

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().grid(column=1, row=0, sticky=(tkinter.N, tkinter.E, tkinter.S, tkinter.W))
        self.figure_canvas.mpl_connect('button_press_event', self.on_click)
        self.figure_canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(column=0, row=1, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S), pady=5)

        ttk.Button(self.controls, text='potrace', command=self.potrace).grid(column=0, row=0)

        self.var_label = tkinter.StringVar(value=labels[0])
        cb = ttk.Combobox(self.controls, textvariable=self.var_label, state='readonly', values=labels, width=8)
        cb.grid(column=1, row=0)
        cb.bind("<<ComboboxSelected>>", self.on_select_piece)

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

        self.var_epsilon = tkinter.DoubleVar(value=0.2)
        ttk.Label(self.controls, text='eps').grid(column=10, row=0)
        ttk.Entry(self.controls, width=5, textvariable=self.var_epsilon).grid(column=11, row=0)

        self.var_offset = tkinter.IntVar(value=0)
        ttk.Label(self.controls, text='offset').grid(column=12, row=0)
        ttk.Entry(self.controls, width=5, textvariable=self.var_offset).grid(column=13, row=0)

        self.potrace_path = None
        self.cursor_xdata = None
        self.render()

    def on_select_piece(self, event):
        print(f"on_select_piece: {self.var_label.get()}")

    def on_click(self, event):
        e = event
        if e.inaxes:
            print(f"on_click: {e.name=} {e.button=} {e.inaxes=} {e.xdata=:.1f} {e.ydata=:.5f}")

    def on_motion(self, event):
        if event.inaxes:
            self.cursor_xdata = event.xdata
            self.update_cursor()
        
    def potrace(self):
        self.potrace_path = puzzler.potrace.piece_to_path(self.piece)
        self.render()

        ds = self.var_ds.get()
        self.curvature = c = Curvature(self.potrace_path, self.var_stepsize.get(), ds=ds)

        other_label = self.labels[1]
        other_piece = None
        for p in self.puzzle.pieces:
            if p.label == other_label:
                other_piece = p
                break
        assert other_piece is not None

        other_potrace_path = puzzler.potrace.piece_to_path(other_piece)
        c2 = Curvature(other_potrace_path, self.var_stepsize.get(), reverse=True, ds=ds)

        eps = self.var_epsilon.get()
        correlation = algorithm_ii(c.curvature, c2.curvature, eps=eps)
        lags = np.arange(-len(c2.curvature),len(c.curvature))

        f = self.figure
        for ax in f.axes.copy():
            f.delaxes(ax)
            
        ax1 = f.add_subplot(3, 1, 1)
        ax2 = f.add_subplot(3, 1, 2)
        ax3 = f.add_subplot(3, 1, 3)

        ax1.plot(np.arange(1,len(c.curvature)+1), c.curvature, np.arange(1,len(c2.curvature)+1), c2.curvature)
        ax1.set_ylabel('curvature')
        ax1.grid(True)

        ax2.plot(lags, correlation)
        ax2.set_ylabel('correlation')
        ax2.set_xlabel('offset')
        ax2.grid(True)

        c1 = c.curvature
        c2 = c2.curvature
        offset = self.var_offset.get()
        if offset < 0:
            start1, end1 = 0, min(len(c1), len(c2)+offset)
        else:
            start1, end1 = offset, min(len(c1)-offset, len(c2))
            
        start2, end2 = start1-offset, end1-offset

        c3 = np.abs(c1[start1:end1] - c2[start2:end2])
        ax3.plot(np.arange(start1,end1), c3)
        ax3.set_ylabel('abs(A - B)')
        ax3.grid(True)

        # self.mc = matplotlib.widgets.MultiCursor(self.figure_canvas, (ax1, ax2, ax3), color='blue', linestyle='--', linewidth=1.)

        self.figure_canvas.draw()

    def get_camera_matrix(self):

        t = puzzler.render.Transform()
        
        canvas_w = int(self.puzzle_canvas.configure('width')[4])
        canvas_h = int(self.puzzle_canvas.configure('height')[4])
        
        camera_matrix = np.array(
            ((1,  0,   0),
             (0, -1, canvas_h-1),
             (0,  0,   1)), dtype=np.float64)
        t.multiply(camera_matrix)

        bbox = self.piece.bbox
        piece_w, piece_h = bbox[1] - bbox[0]

        scale = 0.95 * min(canvas_w / piece_w, canvas_h / piece_h)
        if scale < 1:
            t.scale(scale)

        t.translate(0.5 * (np.array((canvas_w, canvas_h)) - (bbox[0] + bbox[1])))

        return t.matrix

    def render(self):
        canvas = self.puzzle_canvas
        canvas.delete('all')

        r = puzzler.renderer.canvas.CanvasRenderer(canvas)

        r.transform(self.get_camera_matrix())

        if self.var_render_perimeter.get():
            r.draw_lines(self.piece.points, width=1, fill='black')

        if self.potrace_path is not None:

            interp_path = puzzler.potrace.InterpolatePath(self.var_stepsize.get()).apply(self.potrace_path)
            
            if self.var_render_potrace_path.get():
                r.draw_lines(interp_path, fill='cyan', width='1')

            if self.var_render_potrace_lines.get():
                for i in self.potrace_path:
                    if isinstance(i,puzzler.potrace.Line):
                        r.draw_lines(np.array([i.v0, i.v1]), width=3, fill='pink')

            if self.var_render_potrace_points.get():
                r.draw_points(np.array(interp_path), radius=1, fill='green', outline='')

        self.update_cursor()

    def update_cursor(self):

        canvas = self.puzzle_canvas
        canvas.delete('cursor')
        if self.cursor_xdata is None:
            return

        c = self.curvature.interp_point(self.cursor_xdata)
        r = puzzler.renderer.canvas.CanvasRenderer(canvas)
        r.transform(self.get_camera_matrix())
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
