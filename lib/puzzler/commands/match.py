import numpy as np
import puzzler

import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class MatchTk:

    def __init__(self, parent, puzzle, labels):
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.t = np.arange(0, 3, .01)
        self.ax = self.fig.add_subplot()
        self.line, = self.ax.plot(self.t, 2 * np.sin(2 * np.pi * self.t))
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("f(t)")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)  # A tk.DrawingArea.
        self.canvas.draw()
        
        # pack_toolbar=False will make it easier to use a layout manager later on.
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.update()
        
        self.canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        self.canvas.mpl_connect("key_press_event", key_press_handler)
        
        button_quit = tkinter.Button(master=parent, text="Quit", command=parent.destroy)

        slider_update = tkinter.Scale(parent, from_=1, to=5, orient=tkinter.HORIZONTAL,
                                      command=self.update_frequency, label="Frequency[Hz]")

        # Packing order is important. Widgets are processed sequentially and if there
        # is no space left, because the window is too small, they are not displayed.
        # The canvas is rather flexible in its size, so we pack it last which makes
        # sure the UI controls are displayed as long as possible.
        button_quit.pack(side=tkinter.BOTTOM)
        slider_update.pack(side=tkinter.BOTTOM)
        toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    def update_frequency(self, new_val):
        # retrieve frequency
        f = float(new_val)

        # update data
        y = 2 * np.sin(2 * np.pi * f * self.t)
        self.line.set_data(self.t, y)

        # required to update canvas and attached toolbar!
        self.canvas.draw()

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
