import numpy as np

class Light:
    def __init__(self, x, y, z, intensity=1000):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity

    def update_pos(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def get_light(self, X, Y):
        dx = X - self.x
        dy = Y - self.y
        dz = self.z

        r = np.sqrt(dx**2 + dy**2 + dz**2)

        nx, ny, nz = dx / r, dy / r, dz / r

        cos_theta = np.clip(nz, 0, 1)

        E = self.intensity * cos_theta / (r**2 + 1e-9)
        return E
