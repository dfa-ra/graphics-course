import numpy as np


class Scene:
    _instance = None

    Wres, Hres = 60, 60  # разрешение сетки

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.update_grid()

    def add_light(self, light):
        self.light = light

    def update_grid(self):
        x = np.linspace(-self.W / 2, self.W / 2, Scene.Wres)
        y = np.linspace(-self.H / 2, self.H / 2, Scene.Hres)
        self.X, self.Y = np.meshgrid(x, y)
