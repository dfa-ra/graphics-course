import numpy as np


class Scene:
    _instance = None

    Wres, Hres = 60, 60

    def __init__(self, W, H):
        self.W, self.H = W, H
        x = np.linspace(-W / 2, W / 2, Scene.Wres)
        y = np.linspace(-H / 2, H / 2, Scene.Hres)
        self.X, self.Y = np.meshgrid(x, y)

    def add_light(self, light):
        self.light = light


