#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

import renderer

# Доли a,b вдоль (B-A) и (C-A) от A; x=a*|B-A|, y=b*|C-A|; внутри: a>=0, b>=0, a+b<=1.
REPORT_POINTS_LOCAL = [
    (0.10, 0.10),
    (0.25, 0.10),
    (0.40, 0.15),
    (0.15, 0.30),
    (0.20, 0.50),
]

REPORT_PRINT_MODE = "verbose"


def parse_lights(text):
    """6 чисел: x y z I0r I0g I0b. 9 чисел: + Ox Oy Oz (ось источника, |O|→1; cos theta = (s·O)/|s|)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        vals = [float(p) for p in parts if p]
        if len(vals) >= 9:
            out.append(tuple(vals[:9]))
        elif len(vals) >= 6:
            out.append(tuple(vals[:6]))
    return out


class App:
    def __init__(self, root):
        self.root = root
        root.title("Треугольник: освещённость и отчёт")
        root.minsize(520, 400)

        main = ttk.Frame(root, padding=4)
        main.pack(fill=tk.X, anchor=tk.N)

        self.vars = {
            "ax": tk.StringVar(value="0"),
            "ay": tk.StringVar(value="0"),
            "az": tk.StringVar(value="0"),
            "bx": tk.StringVar(value="1"),
            "by": tk.StringVar(value="0"),
            "bz": tk.StringVar(value="0"),
            "cx": tk.StringVar(value="0"),
            "cy": tk.StringVar(value="1"),
            "cz": tk.StringVar(value="0"),
            "grid_w": tk.StringVar(value="50"),
            "grid_h": tk.StringVar(value="50"),
            "pixel": tk.StringVar(value="10"),
            "ox": tk.StringVar(value="0.6"),
            "oy": tk.StringVar(value="0.5"),
            "oz": tk.StringVar(value="1.2"),
            "sr": tk.StringVar(value="0.85"),
            "sg": tk.StringVar(value="0.2"),
            "sb": tk.StringVar(value="0.15"),
            "kd": tk.StringVar(value="0.65"),
            "ks": tk.StringVar(value="0.35"),
            "p": tk.StringVar(value="48"),
        }
        self.report_mode = tk.StringVar(value=REPORT_PRINT_MODE)
        self.show_outside_bg = tk.BooleanVar(value=False)

        left = ttk.Frame(main, padding=(0, 0, 8, 0))
        left.pack(side=tk.LEFT, anchor=tk.N)

        def row(r, label, keys):
            ttk.Label(left, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            for k, c in zip(keys, (1, 2, 3)):
                e = ttk.Entry(left, width=7, textvariable=self.vars[k])
                e.grid(row=r, column=c, padx=1, pady=1)
                e.bind("<KeyRelease>", self._schedule)
                e.bind("<FocusOut>", self._schedule)

        row(0, "A", ("ax", "ay", "az"))
        row(1, "B", ("bx", "by", "bz"))
        row(2, "C", ("cx", "cy", "cz"))
        row(3, "W,H,px", ("grid_w", "grid_h", "pixel"))
        row(4, "Наблюд.", ("ox", "oy", "oz"))
        row(5, "Цвет", ("sr", "sg", "sb"))
        row(6, "kd ks p", ("kd", "ks", "p"))

        ttk.Label(left, text="Свет: x y z I0r I0g I0b [Ox Oy Oz]").grid(row=7, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        self.txt_lights = tk.Text(left, width=34, height=3, wrap="none")
        self.txt_lights.grid(row=8, column=0, columnspan=4, sticky=tk.W)
        self.txt_lights.insert(
            "1.0",
            "2.0 2.0 2.0 1.0 1.0 1.0\n"
            "-1.0 0.5 1.0 0.4 0.5 0.9\n",
        )
        self.txt_lights.bind("<KeyRelease>", self._schedule)
        self.txt_lights.bind("<<Paste>>", lambda e: root.after(10, self._schedule))

        ttk.Checkbutton(
            left,
            text="Показывать фон вне треугольника",
            variable=self.show_outside_bg,
            command=self._schedule,
        ).grid(row=9, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        rf = ttk.Frame(left)
        rf.grid(row=10, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
        ttk.Label(rf, text="Отчёт:").pack(side=tk.LEFT)
        ttk.Radiobutton(rf, text="подробно", variable=self.report_mode, value="verbose").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(rf, text="компактно", variable=self.report_mode, value="compact").pack(side=tk.LEFT)
        ttk.Button(left, text="Печать отчёта в консоль", command=self._print_report).grid(
            row=11, column=0, columnspan=4, sticky=tk.W, pady=4
        )

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, anchor=tk.N)
        self.canvas = tk.Canvas(right, highlightthickness=1, highlightbackground="#bbbbbb", bg=renderer.CANVAS_BG)
        self.canvas.pack()
        ttk.Label(
            right,
            wraplength=520,
            font=("TkDefaultFont", 9),
        ).pack(anchor=tk.W, pady=(4, 0))

        self._job = None
        for v in self.vars.values():
            v.trace_add("write", lambda *a: self._schedule())
        self._schedule()

    def _schedule(self, _evt=None):
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(80, self._redraw)

    def _read_scene(self):
        def f(k):
            return float(self.vars[k].get())

        def fi(k):
            return int(float(self.vars[k].get()))

        lights = parse_lights(self.txt_lights.get("1.0", "end"))
        if not lights:
            lights = [(2.0, 2.0, 2.0, 1.0, 1.0, 1.0)]

        return {
            "ax": f("ax"),
            "ay": f("ay"),
            "az": f("az"),
            "bx": f("bx"),
            "by": f("by"),
            "bz": f("bz"),
            "cx": f("cx"),
            "cy": f("cy"),
            "cz": f("cz"),
            "grid_w": max(1, fi("grid_w")),
            "grid_h": max(1, fi("grid_h")),
            "pixel_size": max(1, fi("pixel")),
            "ox": f("ox"),
            "oy": f("oy"),
            "oz": f("oz"),
            "surf_r": f("sr"),
            "surf_g": f("sg"),
            "surf_b": f("sb"),
            "kd": f("kd"),
            "ks": f("ks"),
            "p": f("p"),
            "lights": lights,
            "report_mode": self.report_mode.get(),
            "show_outside_background": self.show_outside_bg.get(),
        }

    def _redraw(self):
        self._job = None
        try:
            scene = self._read_scene()
        except (ValueError, tk.TclError):
            return
        renderer.render_triangle(self.canvas, scene)

    def _print_report(self):
        try:
            scene = self._read_scene()
        except (ValueError, tk.TclError) as e:
            print("Ошибка параметров:", e)
            return
        scene["report_mode"] = self.report_mode.get()
        pts = REPORT_POINTS_LOCAL
        res = renderer.build_report_results(scene, pts)
        renderer.print_report_values(scene, pts, res)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
