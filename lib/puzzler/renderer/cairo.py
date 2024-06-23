import puzzler.render
import cairo
import math
import numpy as np
import PIL.Image
import PIL.ImageTk
from tkinter import *

class CairoRenderer(puzzler.render.Renderer):

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

    def transform(self, m):
        xx = m[0][0]
        yx = m[1][0]
        xy = m[0][1]
        yy = m[1][1]
        x0 = m[0][2]
        y0 = m[1][2]
        self.context.transform(cairo.Matrix(xx, yx, xy, yy, x0, y0))

    def translate(self, xy):
        self.context.translate(*xy)

    def rotate(self, rad):
        self.context.rotate(rad)

    def user_to_device(self, points):
        return np.array([self.context.user_to_device(*p) for p in points])

    def save(self):
        self.context.save()

    def restore(self):
        self.context.restore()

    def draw_polygon(self, points, fill=None, outline=(0,0,0), width=1, tags=None):

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

    def make_font(self, face, size):

        font = None
        with puzzler.render.save(self):
            ctx = self.context
            ctx.select_font_face(face)
            (w, h) = ctx.device_to_user_distance(size, size)
            ctx.set_font_matrix(cairo.Matrix(xx=w, yy=h))
            font = ctx.get_scaled_font()
        return font

    def draw_text(self, xy, text, font=None, fill=(0, 0, 0)):

        ctx = self.context
        ctx.save()

        if font:
            ctx.set_scaled_font(font)
        else:
            font = ctx.get_scaled_font()

        extents = font.text_extents(text)
        # print(f"draw_text: text=\"{text}\" {extents=}")

        xy = xy - np.array((extents.width, extents.height)) * .5

        ctx.move_to(*xy)

        if fill:
            ctx.set_source_rgb(*self.get_color(fill))
        
        ctx.show_text(text)

        ctx.restore()

    def commit(self):
        surface = self.surface
        surface.flush()

        if True:
            w, h = surface.get_width(), surface.get_height()
            stride = surface.get_stride()
            ystep = 1
            image = PIL.Image.frombuffer('RGBA', (w,h), surface.get_data().tobytes(),
                                         'raw', 'BGRA', stride, ystep)
            # image.save("yuck.png")
            displayed_image = PIL.ImageTk.PhotoImage(image=image)
        else:
            surface.write_to_png('fnord.png')
            displayed_image = PhotoImage(file='fnord.png')
            
        self.canvas.create_image((0, 0), image=displayed_image, anchor=NW)
        return displayed_image

