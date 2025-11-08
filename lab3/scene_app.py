import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from light import Light
from scene import Scene


class SceneApp:

    def __init__(self, root):
        self.scene = Scene(2000, 2000)
        light = Light(500, 500, 1000, intensity=1000)
        self.scene.add_light(light)

        self.frame_controls = tk.Frame(root)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(self.frame_controls, text="X источника").pack()
        self.x_slider = tk.Scale(self.frame_controls, from_=-1000, to=1000,
                                 orient='horizontal', command=self.update_plot)
        self.x_slider.set(500)
        self.x_slider.pack(fill='x')

        tk.Label(self.frame_controls, text="Y источника").pack()
        self.y_slider = tk.Scale(self.frame_controls, from_=-1000, to=1000,
                                 orient='horizontal', command=self.update_plot)
        self.y_slider.set(500)
        self.y_slider.pack(fill='x')

        tk.Label(self.frame_controls, text="Z источника").pack()
        self.z_slider = tk.Scale(self.frame_controls, from_=100, to=2000,
                                 orient='horizontal', command=self.update_plot)
        self.z_slider.set(1000)
        self.z_slider.pack(fill='x')

        tk.Label(self.frame_controls, text="Сила света").pack()
        self.intensity_slider = tk.Scale(self.frame_controls, from_=0, to=5000,
                                         orient='horizontal', command=self.update_plot)
        self.intensity_slider.set(1000)
        self.intensity_slider.pack(fill='x')

        tk.Label(self.frame_controls, text="Ширина сцены (W)").pack()
        self.W_slider = tk.Scale(self.frame_controls, from_=500, to=4000,
                                 orient='horizontal', command=self.update_scene_size)
        self.W_slider.set(self.scene.W)
        self.W_slider.pack(fill='x')

        tk.Label(self.frame_controls, text="Высота сцены (H)").pack()
        self.H_slider = tk.Scale(self.frame_controls, from_=500, to=4000,
                                 orient='horizontal', command=self.update_scene_size)
        self.H_slider.set(self.scene.H)
        self.H_slider.pack(fill='x')

        # === 3D ВИЗУАЛИЗАЦИЯ ===
        self.frame_plot = tk.Frame(root)
        self.frame_plot.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        fig = plt.Figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()  # первичная отрисовка
        root.mainloop()

    def update_scene_size(self, *args):
        self.scene.W = self.W_slider.get()
        self.scene.H = self.H_slider.get()
        self.scene.update_grid()  # пересоздаём X, Y
        self.update_plot()

    def update_plot(self, *args):
        xL = self.x_slider.get()
        yL = self.y_slider.get()
        zL = self.z_slider.get()
        intensity = self.intensity_slider.get()

        E = self.light_update(xL, yL, zL, intensity)

        self.ax.cla()

        E_scaled = E * 1e4

        self.ax.plot_surface(self.scene.X, self.scene.Y, np.zeros_like(E),
                             color='gray', alpha=0.3)

        self.ax.plot_surface(self.scene.X, self.scene.Y, E_scaled,
                             facecolors=plt.cm.inferno(E / np.max(E + 1e-9)),
                             rstride=1, cstride=1, shade=False)

        self.ax.scatter(self.scene.light.x, self.scene.light.y, self.scene.light.z,
                        color='yellow', s=80, edgecolors='k', label='Источник света')

        self.ax.set_xlim(-self.scene.W / 2, self.scene.W / 2)
        self.ax.set_ylim(-self.scene.H / 2, self.scene.H / 2)

        z_max = max(np.max(E_scaled), self.scene.light.z)
        self.ax.set_zlim(0, z_max * 1.1)

        self.ax.set_xlabel('X (мм)')
        self.ax.set_ylabel('Y (мм)')
        self.ax.set_zlabel('Освещённость')
        self.ax.set_title('3D сцена освещенности')
        self.ax.legend(loc='upper right')

        self.canvas.draw()

    def light_update(self, xL, yL, zL, intensity):
        self.scene.light.update_pos(xL, yL, zL)
        self.scene.light.intensity = intensity
        E = self.scene.light.get_light(self.scene.X, self.scene.Y)
        return E

