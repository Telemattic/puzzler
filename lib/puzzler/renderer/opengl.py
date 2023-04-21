import puzzler.render
import numpy as np
from OpenGL import GL
import OpenGL.GL.shaders
import ctypes
from ctypes import WinDLL, c_void_p
from ctypes.wintypes import HDC
from OpenGL.WGL import PIXELFORMATDESCRIPTOR, ChoosePixelFormat, \
    SetPixelFormat, SwapBuffers, wglCreateContext, wglMakeCurrent
import pyopengltk
import tkinter as tk

class OpenGLFrame(tk.Frame):

    def __init__(self, *args, **kw):
        # Set background to empty string to avoid
        # flickering overdraw by Tk
        kw['bg'] = ""
        tk.Frame.__init__(self, *args, **kw)
        self._window = None
        self._context = None
        self.bind('<Map>', self.tkMap)

    def tkMap(self, evt):
        """" Called when frame goes onto the screen """
        _user32 = WinDLL('user32')
        GetDC = _user32.GetDC
        GetDC.restype = HDC
        GetDC.argtypes = [c_void_p]

        self._window = GetDC(self.winfo_id())
        
        pfd = PIXELFORMATDESCRIPTOR()
        PFD_TYPE_RGBA =         0
        PFD_MAIN_PLANE =        0
        PFD_DOUBLEBUFFER =      0x00000001
        PFD_DRAW_TO_WINDOW =    0x00000004
        PFD_SUPPORT_OPENGL =    0x00000020
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        pfd.iPixelType = PFD_TYPE_RGBA
        pfd.cColorBits = 24
        pfd.cDepthBits = 0
        pfd.iLayerType = PFD_MAIN_PLANE
        
        pixelformat = ChoosePixelFormat(self._window, pfd)
        SetPixelFormat(self._window, pixelformat, pfd)
        self._context = wglCreateContext(self._window)
        wglMakeCurrent(self._window, self._context)
        
        self.unbind('<Map>')

    def tkMakeCurrent(self):
        if self.winfo_ismapped():
            wglMakeCurrent(self._window, self._context)

    def tkSwapBuffers(self):
        if self.winfo_ismapped():
            SwapBuffers(self._window)

class OpenGLRenderer(puzzler.render.Renderer):

    def __init__(self, frame):
        # print(f"OpenGLRenderer: {frame._window=} {frame._context=}")
        self.frame = frame
        w, h = frame.winfo_width(), frame.winfo_height()
        self.frame.tkMakeCurrent()
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(1., 1., 1., 1.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self._colors = dict()

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, w, h, 0, -1, 1)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)

    def transform(self, m3):

        m4 = np.zeros((4, 4), dtype=np.float32)
        m4[0][0] = m3[0][0]
        m4[0][1] = m3[0][1]
        m4[1][0] = m3[1][0]
        m4[1][1] = m3[1][1]
        m4[3][0] = m3[0][2]
        m4[3][1] = m3[1][2]
        m4[3][3] = 1.

        # print(f"transform: {m3=} {m4=}")

        GL.glMultMatrixf(m4.tobytes())

    def translate(self, xy):

        GL.glTranslatef(xy[0], xy[1], 0.)

    def rotate(self, rad):
        c, s = np.cos(rad), np.sin(rad)
        m = np.array(((c, -s, 0),
                      (s,  c, 0),
                      (0,  0, 1)))
        self.transform(m)

    def user_to_device(self, points):
        raise NotImplementedError

    def save(self):
        GL.glPushMatrix()

    def restore(self):
        GL.glPopMatrix()

    def draw_polygon(self, points, fill=None, outline=(0,0,0), width=1):

        if outline:
            GL.glColor3fv(self.get_color(outline))
            GL.glLineWidth(width)
            self.draw_arrays(GL.GL_LINE_LOOP, points)

        if fill:
            raise NotImplementedError

    def get_color(self, color):
        
        if isinstance(color,str):
            if x := self._colors.get(color):
                return x
            x = tuple((c / 65535 for c in self.frame.winfo_rgb(color)))
            self._colors[color] = x
            return x

        return color

    def draw_lines(self, points, fill=(0, 0, 0), width=1):

        GL.glColor3fv(self.get_color(fill))
        GL.glLineWidth(width)

        self.draw_arrays(GL.GL_LINES, points)

    def draw_ellipse(self, center, semi_major, semi_minor, phi, fill=None, outline=(0, 0, 0), width=1):

        ellipse = puzzler.geometry.Ellipse(center, semi_major, semi_minor, phi)
        points = puzzler.geometry.get_ellipse_points(ellipse, npts=40)

        if fill:
            GL.glColor3fv(self.get_color(fill))
            points2 = np.vstack((ellipse.center, points))
            self.draw_arrays(GL.GL_TRIANGLE_FAN, points2)

        if outline:
            GL.glColor3fv(self.get_color(outline))
            GL.glLineWidth(width)
            self.draw_arrays(GL.GL_LINE_LOOP, points)

    def draw_arrays(self, mode, points):
        
        gltype = None
        if points.dtype == np.float64:
            gltype = GL.GL_DOUBLE
        elif points.dtype == np.float32:
            gltype = GL.GL_FLOAT
        elif points.dtype == np.int32:
            gltype = GL.GL_INT
        elif points.dtype == np.int16:
            gltype = GL.GL_SHORT
        else:
            assert gltype is not None, f"unsupported dtype={points.dtype}"
        
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(2, gltype, 0, points.tobytes())
        GL.glDrawArrays(mode, 0, len(points))
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
    
    def make_font(self, face, size):

        return "not a font"
        raise NotImplementedError

    def draw_text(self, xy, text, font=None, fill=(0, 0, 0)):

        return
        raise NotImplementedError

    def commit(self):

        self.frame.tkSwapBuffers()


