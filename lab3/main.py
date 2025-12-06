# lr3_akg_perfect_square_pixels.py
# Тепловая карта с квадратными пикселями и динамическим размером

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk

class LightingLabPerfect:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР№3 — Квадратные пиксели")
        self.root.geometry("1500x950")

        # Параметры источника и количество пикселей
        self.vars = {
            'xL': tk.DoubleVar(value=624.0),
            'yL': tk.DoubleVar(value=0.0),
            'zL': tk.DoubleVar(value=1000.0),
            'I0': tk.DoubleVar(value=1000.0),
            'W':  tk.IntVar(value=600),   # пикселей по X
            'H':  tk.IntVar(value=600),   # пикселей по Y
            'R':  tk.DoubleVar(value=800.0)
        }

        # Размер базовой сцены (1 пиксель = 1 мм для начального масштаба)
        self.base_scene_size = 2000.0

        self.create_widgets()
        self.setup_plot()

        for var in self.vars.values():
            var.trace_add("write", lambda *args: self.root.after_idle(self.update_plot))

        self.update_plot()

    def create_widgets(self):
        left = ttk.Frame(self.root, padding="15")
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Параметры", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,15))

        params = [
            ("X источника, мм", 'xL', -5000, 5000),
            ("Y источника, мм", 'yL', -5000, 5000),
            ("Z источника, мм", 'zL', 100, 10000),
            ("Сила света I₀, Вт/ср", 'I0', 10, 20000),
            ("Пикселей по ширине W", 'W', 100, 1200),
            ("Пикселей по высоте H", 'H', 100, 1200),
            ("Радиус круга R, мм", 'R', 50, 6000)
        ]

        self.labels = {}
        for i, (txt, key, mn, mx) in enumerate(params, 1):
            ttk.Label(left, text=txt).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Scale(left, from_=mn, to=mx, variable=self.vars[key], length=230).grid(row=i, column=1, padx=8, pady=4)
            lbl = ttk.Label(left, text="", width=12)
            lbl.grid(row=i, column=2)
            self.labels[key] = lbl

            if isinstance(self.vars[key], tk.IntVar):
                self.vars[key].trace_add("write", lambda *_, k=key: self.labels[k].config(text=str(int(self.vars[k].get()))))
            else:
                self.vars[key].trace_add("write", lambda *_, k=key: self.labels[k].config(text=f"{self.vars[k].get():.1f}"))

        ttk.Button(left, text="Сохранить PNG", command=self.save).grid(row=len(params)+1, column=0, columnspan=3, pady=20)

    def setup_plot(self):
        self.fig = Figure(figsize=(14, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def calculate(self):
        p = {k: v.get() for k, v in self.vars.items()}
        Wres = int(p['W'])
        Hres = int(p['H'])

        # Пропорциональный физический размер сцены, чтобы пиксели были квадратными
        scene_W = self.base_scene_size * (Wres / 600)  # базовое количество пикселей 600
        scene_H = self.base_scene_size * (Hres / 600)

        # координаты центров пикселей
        x = np.linspace(-scene_W/2, scene_W/2, Wres, endpoint=False) + scene_W/(2*Wres)
        y = np.linspace(-scene_H/2, scene_H/2, Hres, endpoint=False) + scene_H/(2*Hres)
        X, Y = np.meshgrid(x, y)

        dx = X - p['xL']
        dy = Y - p['yL']
        dz = p['zL']
        r = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-12)
        cos_theta = np.clip(dz / r, 0, 1)
        E = p['I0'] * cos_theta / (r**2)

        mask = X**2 + Y**2 <= p['R']**2

        # Пиксельное изображение
        E_norm = np.clip(E / np.nanmax(E) * 255, 0, 255).astype(np.uint8)
        rgb = plt.cm.hot(E_norm / 255.0)[:, :, :3]
        E_img = np.zeros_like(rgb)
        E_img[mask] = rgb[mask]
        E_img[~mask] = 0

        # Статистика
        stats = {
            'center': E[Hres//2, Wres//2],
            'max': np.max(E[mask]) if np.any(mask) else 0,
            'min': np.min(E[mask]) if np.any(mask) else 0,
            'mean': np.mean(E[mask]) if np.any(mask) else 0
        }

        def val_at(xx, yy):
            xi = int((xx + scene_W/2) / scene_W * Wres)
            yi = int((yy + scene_H/2) / scene_H * Hres)
            if 0 <= xi < Wres and 0 <= yi < Hres:
                return E[yi, xi]
            return np.nan
        stats['edge_x'] = val_at(p['R'], 0)
        stats['edge_y'] = val_at(0, p['R'])

        return E_img, E, x, y, stats, p, Wres, Hres, scene_W, scene_H

    def update_plot(self):
        self.fig.clear()
        E_img, E_full, x_line, y_line, stats, p, Wres, Hres, scene_W, scene_H = self.calculate()

        # Тепловая карта
        ax1 = self.fig.add_subplot(2, 2, (1,2))
        ax1.imshow(E_img, extent=[-scene_W/2, scene_W/2, -scene_H/2, scene_H/2],
                   origin='lower', interpolation='none')
        circle = plt.Circle((0, 0), p['R'], color='cyan', fill=False, lw=2, ls='--')
        ax1.add_patch(circle)
        ax1.plot(p['xL'], p['yL'], 'yellow', marker='*', markersize=16, markeredgecolor='black', mew=1)
        ax1.set_title(f'Тепловая карта {Wres}×{Hres} пикселей')
        ax1.set_xlabel('X, мм')
        ax1.set_ylabel('Y, мм')
        ax1.set_aspect('equal', adjustable='box')

        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 255))
        self.fig.colorbar(sm, ax=ax1, label='Нормированная яркость (0–255)', shrink=0.8)

        # Сечение по X
        ax2 = self.fig.add_subplot(2, 2, 3)
        ax2.plot(x_line, E_full[Hres//2, :], color='#1f77b4', lw=1.8)
        ax2.set_xlim(-scene_W/2, scene_W/2)
        ax2.set_title('Сечение по X (Y = 0)')
        ax2.set_xlabel('X, мм')
        ax2.set_ylabel('E, лк')
        ax2.grid(True, alpha=0.3)

        # Сечение по Y
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.plot(y_line, E_full[:, Wres//2], color='#ff7f0e', lw=1.8)
        ax3.set_xlim(-scene_H/2, scene_H/2)
        ax3.set_title('Сечение по Y (X = 0)')
        ax3.set_xlabel('Y, мм')
        ax3.set_ylabel('E, лк')
        ax3.grid(True, alpha=0.3)

        txt = (f"Источник: ({p['xL']:.1f}, {p['yL']:.1f}, {p['zL']:.0f}) мм | I₀ = {p['I0']:.0f} Вт/ср\n"
               f"Радиус R = {p['R']:.0f} мм\n"
               f"Разрешение: W×H = {Wres}×{Hres} пикселей\n"
               f"Физический размер сцены: {scene_W:.0f}×{scene_H:.0f} мм\n\n"
               f"E(0,0) = {stats['center']:.2f} лк | E(R,0) = {stats['edge_x']:.2f} лк | E(0,R) = {stats['edge_y']:.2f} лк\n"
               f"Eₘₐₓ = {stats['max']:.2f} лк | Eₘᵢₙ = {stats['min']:.2f} лк | Eₛᵣ = {stats['mean']:.2f} лк")

        self.fig.suptitle("ЛР№3 — Освещённость от точечного источника", fontsize=14)
        self.fig.text(0.01, 0.01, txt, fontsize=10.5, va='bottom',
                      bbox=dict(boxstyle="round,pad=0.7", facecolor="#f0f0f0"))

        self.fig.tight_layout(rect=[0, 0.14, 1, 0.94])
        self.canvas.draw()

    def save(self):
        from datetime import datetime
        fn = f"LR3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.fig.savefig(fn, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Сохранено: {fn}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LightingLabPerfect(root)
    root.mainloop()
