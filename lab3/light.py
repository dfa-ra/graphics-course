import numpy as np


class Light:
    I0 = 1000

    def __init__(self, X, Y, Z):
        self.x = X
        self.y = Y
        self.z = Z

    def get_light(self, X, Y):
        r = np.sqrt((X - self.x) ** 2 + (Y - self.y) ** 2 + self.z ** 2)
        E = Light.I0 * self.z / (r ** 3)
        E_norm = E / np.max(E)
        return E_norm

    def update_pos(self, X, Y, Z):
        self.x = X
        self.y = Y
        self.z = Z