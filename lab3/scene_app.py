import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec

from light import Light
from scene import Scene


class SceneApp:

    def __init__(self, root):
        # --- Инициализация сцены и источника света ---
        self.scene = Scene(2000, 2000)
        light = Light(0, 0, 1000, intensity=1000, spot_radius=800)
        self.scene.add_light(light)

        # --- Панель управления ---
        self.frame_controls = tk.Frame(root, bg="#ddd")
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        labels = [
            ("X источника", (-1000, 1000, 0), "x_slider"),
            ("Y источника", (-1000, 1000, 0), "y_slider"),
            ("Z источника", (100, 2000, 1000), "z_slider"),
            ("Сила света", (0, 5000, 1000), "intensity_slider"),
            ("Ширина сцены (W)", (500, 4000, 2000), "W_slider"),
            ("Высота сцены (H)", (500, 4000, 2000), "H_slider"),
            ("Радиус прожектора", (100, 4000, 800), "spot_slider"),
        ]

        for label, (vmin, vmax, vinit), attr in labels:
            tk.Label(self.frame_controls, text=label, bg="#ddd").pack()
            if attr in ("W_slider", "H_slider"):
                command = self.update_scene_size
            else:
                command = self.update_plot

            scale = tk.Scale(
                self.frame_controls,
                from_=vmin,
                to=vmax,
                orient="horizontal",
                command=command,
                length=200,
                bg="#ddd"
            )
            scale.set(vinit)
            scale.pack(fill="x", pady=2)
            setattr(self, attr, scale)

        # --- Рамка для графиков ---
        self.frame_plot = tk.Frame(root)
        self.frame_plot.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # --- Создаем фигуру с 3 подграфиками: сверху 3D и тепловая карта, снизу сечения ---
        self.fig = plt.Figure(figsize=(10, 9))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1])

        self.ax_heat = self.fig.add_subplot(gs[0])             # тепловая карта
        self.ax3d = self.fig.add_subplot(gs[1], projection='3d')  # 3D сцена
        self.ax_cut = self.fig.add_subplot(gs[2])              # сечения

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Первичная отрисовка ---
        self.update_plot()
        root.mainloop()

    def update_scene_size(self, *args):
        """Обновление размеров сцены"""
        self.scene.Wres = self.scene.Wres * self.W_slider.get() / self.scene.W
        self.scene.Hres = self.scene.Hres * self.H_slider.get() / self.scene.H
        # print(self.scene.Hres)
        self.scene.W = self.W_slider.get()
        self.scene.H = self.H_slider.get()

        self.scene.update_grid()
        self.update_plot()

    def light_update(self, xL, yL, zL, intensity, spot_radius):
        """Обновление параметров источника света"""
        self.scene.light.update_pos(xL, yL, zL)
        self.scene.light.intensity = intensity
        self.scene.light.spot_radius = spot_radius
        E = self.scene.light.get_light(self.scene.X, self.scene.Y)
        return E

    def update_plot(self, *args):
        """Перерисовка всей сцены и графиков"""
        xL = self.x_slider.get()
        yL = self.y_slider.get()
        zL = self.z_slider.get()
        intensity = self.intensity_slider.get()
        spot_radius = self.spot_slider.get()

        E = self.light_update(xL, yL, zL, intensity, spot_radius)

        # === Очистка всех осей ===
        self.ax3d.cla()
        self.ax_cut.cla()
        self.ax_heat.cla()

        # === (1) ТЕПЛОВАЯ КАРТА (вид сверху) ===
        im = self.ax_heat.imshow(
            E,
            extent=[-self.scene.W / 2, self.scene.W / 2, -self.scene.H / 2, self.scene.H / 2],
            origin='lower',
            cmap='gray',
            interpolation='nearest',
            aspect='equal'
        )

        theta = np.linspace(0, 2 * np.pi, 200)
        circle_x = self.scene.light.x + spot_radius * np.cos(theta)
        circle_y = self.scene.light.y + spot_radius * np.sin(theta)
        self.ax_heat.plot(circle_x, circle_y, 'c--', linewidth=1.5, label='Граница прожектора')

        self.ax_heat.set_xlim(-self.scene.W / 2, self.scene.W / 2)
        self.ax_heat.set_ylim(-self.scene.H / 2, self.scene.H / 2)
        self.ax_heat.set_title("Тепловая карта освещённости (вид сверху)")

        # === (2) 3D СЦЕНА ===
        self.ax3d.plot_surface(
            self.scene.X, self.scene.Y, E,
            facecolors=plt.cm.inferno(E / np.max(E + 1e-9)),
            rstride=1, cstride=1, shade=False
        )
        self.ax3d.scatter(
            self.scene.light.x, self.scene.light.y, self.scene.light.z,
            color='yellow', s=80, edgecolors='k', label='Источник света'
        )
        self.ax3d.plot(circle_x, circle_y, np.zeros_like(theta), 'c--', label='Граница прожектора')

        self.ax3d.set_xlim(-self.scene.W / 2, self.scene.W / 2)
        self.ax3d.set_ylim(-self.scene.H / 2, self.scene.H / 2)
        self.ax3d.set_zlim(0, np.max(E) * 1.1)
        self.ax3d.set_xlabel('X (мм)')
        self.ax3d.set_ylabel('Y (мм)')
        self.ax3d.set_zlabel('E (Вт/мм²)')
        self.ax3d.set_title('3D сцена освещенности')
        self.ax3d.legend()

        # === (3) ГРАФИКИ СЕЧЕНИЙ ===
        Ny, Nx = E.shape
        mid_y = Ny // 2
        mid_x = Nx // 2

        x_line = self.scene.X[mid_y, :]
        y_line = self.scene.Y[:, mid_x]

        self.ax_cut.plot(x_line, E[mid_y, :], label='Сечение по X (центр Y)')
        self.ax_cut.plot(y_line, E[:, mid_x], label='Сечение по Y (центр X)')
        self.ax_cut.set_xlim(-self.scene.W / 2, self.scene.W / 2)
        self.ax_cut.set_ylim(0, np.max(E) * 1.1)
        self.ax_cut.set_xlabel('Координата (мм)')
        self.ax_cut.set_ylabel('Освещённость E (Вт/мм²)')
        self.ax_cut.set_title('Графики сечений по центру')
        self.ax_cut.legend()
        self.ax_cut.grid(True)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
