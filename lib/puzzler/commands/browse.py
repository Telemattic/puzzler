import cairo
import cv2 as cv
import PIL.Image
import PIL.ImageTk
import math
import numpy as np
import puzzler
import re

from tkinter import *
from tkinter import font
from tkinter import ttk

# Camera = puzzler.commands.align.Camera
# MoveCamera = puzzler.commands.align.MoveCamera

class CairoRenderer:

    def __init__(self, canvas):
        self.canvas  = canvas
        w, h = canvas.winfo_width(), canvas.winfo_height()
        # print(f"CairoRenderer: {w=} {h=}")
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
        self.context = cairo.Context(self.surface)
        self._colors = dict()

        ctx = self.context

        ctx.save()
        ctx.rectangle(0, 0, w, h)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()
        ctx.restore()

    def draw_polygon(self, points, fill=None, outline=(0,0,0), width=1):

        ctx = self.context

        ctx.save()

        if outline and width:
            (w, h) = ctx.device_to_user_distance(width, width)
            ctx.set_line_width(math.fabs(w))
        
        ctx.move_to(*points[-1])
        for p in points:
            ctx.line_to(*p)
        ctx.close_path()
        
        if fill:
            ctx.set_source_rgb(*self.get_color(fill))
            if outline:
                ctx.fill_preserve()
            else:
                ctx.fill()
                
        if outline:
            ctx.set_source_rgb(*self.get_color(outline))
            ctx.stroke()
        
        ctx.restore()

    def get_color(self, color):
        
        if isinstance(color,str):
            if x := self._colors.get(color):
                return x
            x = tuple((c / 65535 for c in self.canvas.winfo_rgb(color)))
            self._colors[color] = x
            return x

        return color

    def draw_lines(self, points, fill=(0, 0, 0), width=1):

        ctx = self.context
        ctx.save()

        if fill and width:
            (w, h) = ctx.device_to_user_distance(width, width)
            ctx.set_line_width(math.fabs(w))
            
        if fill:
            ctx.set_source_rgb(*self.get_color(fill))
        
        def pairwise(x):
            i = iter(x)
            return zip(i, i)

        for p1, p2 in pairwise(points):
            ctx.move_to(*p1)
            ctx.line_to(*p2)

        ctx.stroke()

        ctx.restore()

    def draw_ellipse(self, center, semi_major, semi_minor, phi, fill=None, outline=(0, 0, 0), width=1):
        
        ctx = self.context
        ctx.save()
        
        if outline and width:
            (w, h) = ctx.device_to_user_distance(width, width)
            ctx.set_line_width(math.fabs(w))
        
        ctx.translate(*center)
        ctx.rotate(phi)
        ctx.scale(semi_major, semi_minor)
        ctx.arc(0., 0., 1., 0, 2 * math.pi)

        if fill:
            ctx.set_source_rgb(*self.get_color(fill))
            if outline:
                ctx.fill_preserve()
            else:
                ctx.fill()
                
        if outline:
            ctx.set_source_rgb(*self.get_color(outline))
            ctx.stroke()

        ctx.restore()

    def draw_text(self, xy, text, font=None, fill=(0, 0, 0)):

        ctx = self.context
        ctx.save()
        
        ctx.select_font_face("Courier New")
        (w, h) = ctx.device_to_user_distance(18, 18)
        ctx.set_font_matrix(cairo.Matrix(xx=w, yy=h))
        
        ctx.move_to(*xy)

        if fill:
            ctx.set_source_rgb(*self.get_color(fill))
        
        ctx.show_text(text)

        ctx.restore()

    def commit(self):
        surface = self.surface

        if True:
            w, h = surface.get_width(), surface.get_height()
            stride = surface.get_stride()
            print(f"surface: {w=} {h=} {stride=}")
            ystep = 1
            image = PIL.Image.frombuffer('RGBA', (w,h), surface.get_data().tobytes(), 'raw', 'BGRA', stride, ystep)
            # image.save("yuck.png")
            displayed_image = PIL.ImageTk.PhotoImage(image=image)
        else:
            surface.write_to_png('fnord.png')
            displayed_image = PhotoImage(file='fnord.png')
            
        self.canvas.create_image((0, 0), image=displayed_image, anchor=NW)
        return displayed_image

class Browser:

    class Outline:

        def __init__(self, piece, epsilon=5):

            assert piece.points is not None
            approx  = cv.approxPolyDP(piece.points, epsilon, True)
            self.poly = np.squeeze(approx)
            self.poly = np.concatenate((self.poly, self.poly[:1,:]))

            ll = np.min(self.poly, 0)
            ur = np.max(self.poly, 0)
            self.bbox   = tuple(ll.tolist() + ur.tolist())
            self.piece  = piece
            self.tag    = None

    def __init__(self, puzzle, use_cairo):

        self.use_cairo = use_cairo
        
        pieces = []
        for p in puzzle.pieces:
            label = p.label
            m = re.fullmatch("^(\w+)(\d+)", label)
            if m:
                pieces.append((m[1], int(m[2]), p))
            else:
                pieces.append((label, None, p))
        pieces.sort()

        self.outlines = [Browser.Outline(p[2]) for p in pieces]

        bbox_w = max(o.bbox[2] - o.bbox[0] for o in self.outlines)
        bbox_h = max(o.bbox[3] - o.bbox[1] for o in self.outlines)

        max_w = 2500 # 1400
        max_h = 1500 # 800

        def compute_rows(cols):
            tile_w = max_w // cols
            tile_h = tile_w * bbox_h // bbox_w
            rows   = max_h // tile_h
            print(f"{cols=} {tile_w=} {tile_h=} {rows=}")
            return rows
            
        cols = 1
        while cols * compute_rows(cols) < len(self.outlines):
            cols += 1

        self.cols = cols
        self.rows = compute_rows(cols)

        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.tile_w = max_w // cols
        self.tile_h = self.tile_w * bbox_h // bbox_w
        self.scale  = min(self.tile_w / bbox_w, self.tile_h / bbox_h)
        self.width  = self.tile_w * self.cols
        self.height = self.tile_h * self.rows

        self.font = None

        print(f"scale={self.scale:.3f}")

    def render(self, canvas, camera):

        if self.use_cairo:
            return self.render_cairo(canvas, camera)

        r = puzzler.render.Renderer(canvas)
        r.transform.multiply(camera.matrix)
        # r.transform.scale(self.scale)

        if not self.font:
            self.font = font.Font(family='Courier', name='pieceLabelFont', size=12)

        for i, o in enumerate(self.outlines):
            x = (i %  self.cols)
            y = (self.rows - 1 - (i // self.cols))
            tx = (x + .5) * self.bbox_w
            ty = (y + .5) * self.bbox_h
            with puzzler.render.save_matrix(r.transform):
                r.transform.translate((tx, ty))
                self.render_outline(r, o)

    def render_outline(self, r, o):

        p = o.piece

        # want the corners of the outline bbox centered within the tile
        bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2

        with puzzler.render.save_matrix(r.transform):
            r.transform.translate(-bbox_center)

            if p.tabs is not None:
                for tab in p.tabs:
                    e = tab.ellipse
                    r.draw_ellipse(e.center, e.semi_major, e.semi_minor, e.phi, fill='cyan', outline='')

            if p.edges is not None and False:
                for edge in p.edges:
                    r.draw_lines(edge.line.pts, width=4, fill='pink')

            r.draw_polygon(o.poly, outline='black', fill='', width=1)

        r.draw_text(np.zeros(2), text=p.label, font=self.font, fill='black')

    def render_cairo(self, canvas, camera):

        r = CairoRenderer(canvas)

        m = camera.matrix
        xx = m[0][0]
        yx = m[1][0]
        xy = m[0][1]
        yy = m[1][1]
        x0 = m[0][2]
        y0 = m[1][2]
        m = cairo.Matrix(xx, yx, xy, yy, x0, y0)

        r.context.transform(m)

        w, h = canvas.winfo_width(), canvas.winfo_height()
        for d in [(0,h), (w//2, h//2), (w,0)]:
            u = r.context.device_to_user(*d)
            print(f"  device={d} user={u}")

        if not self.font:
            self.font = font.Font(family='Courier', name='pieceLabelFont', size=12)

        for i, o in enumerate(self.outlines):
            x = (i %  self.cols)
            y = (self.rows - 1 - (i // self.cols))
            tx = (x + .5) * self.bbox_w
            ty = (y + .5) * self.bbox_h

            r.context.save()
            r.context.translate(tx, ty)
            self.render_outline_cairo(r, o)
            r.context.restore()

        return r.commit()

    def render_outline_cairo(self, r, o):

        p = o.piece

        # want the corners of the outline bbox centered within the tile
        bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2

        r.context.translate(*-bbox_center)

        if p.tabs is not None:
            for tab in p.tabs:
                e = tab.ellipse
                r.draw_ellipse(e.center, e.semi_major, e.semi_minor, e.phi, fill='cyan', outline=None)
                
        if p.edges is not None:
            for edge in p.edges:
                r.draw_lines(edge.line.pts, width=4, fill='pink')

        r.draw_polygon(o.poly, outline='black', fill=None)

        r.draw_text(np.zeros(2), text=p.label, fill='black')

class BrowseTk:

    def __init__(self, parent, puzzle, use_cairo=False):

        self.browser = Browser(puzzle, use_cairo)

        w, h = self.browser.width, self.browser.height

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = Canvas(self.frame, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))

        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<Configure>", self.canvas_configure)

        center = np.array((w / 2, h / 2), dtype=np.float64)
        zoom = self.browser.scale
        viewport = (w, h)
        
        self.camera = puzzler.commands.align.Camera(center, zoom, viewport)
        self.draggable = None

    def canvas_configure(self, event):
        print(f"{event=}")
        self.camera.viewport = (event.width, event.height)
        self.render()

    def mouse_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.render()

    def canvas_press(self, event):

        self.draggable = puzzler.commands.align.MoveCamera(self.camera)
        self.draggable.start(np.array((event.x, event.y)))
        self.render()
        
    def canvas_drag(self, event):

        if self.draggable:
            self.draggable.drag(np.array((event.x, event.y)))
            self.render()

    def canvas_release(self, event):
                
        if self.draggable:
            self.draggable.commit()
            self.draggable = None
            self.render()

    def render(self):

        self.canvas.delete('all')

        self.displayed_image = self.browser.render(self.canvas, self.camera)

def browse(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    root = Tk()
    ui = BrowseTk(root, puzzle, args.cairo)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: browse")
    root.mainloop()

def add_parser(commands):
    parser_browse = commands.add_parser("browse", help="browse pieces")
    parser_browse.add_argument("-c", "--cairo", action='store_const', const=True, default=False,
                               help="use cairo rendering (default: tk)")
    parser_browse.set_defaults(func=browse)
