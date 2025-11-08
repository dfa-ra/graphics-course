import numpy as np


class Light:
    def __init__(self, x, y, z, intensity=1000, cone_angle=None, spot_radius=None):
        """
        x, y, z — координаты источника (мм)
        intensity — сила света
        cone_angle — угол прожектора (в градусах) от вертикали (если None — светит во все стороны)
        spot_radius — радиус освещаемой области (мм) на плоскости (если None — без ограничений)
        """
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.cone_angle = cone_angle
        self.spot_radius = spot_radius

    def update_pos(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def set_spot(self, cone_angle=None, spot_radius=None):
        """Изменить параметры прожектора"""
        self.cone_angle = cone_angle
        self.spot_radius = spot_radius

    def get_light(self, X, Y):

        X_m = X / 1000
        Y_m = Y / 1000
        xL_m = self.x / 1000
        yL_m = self.y / 1000
        zL_m = self.z / 1000

        dx = X_m - xL_m
        dy = Y_m - yL_m
        dz = zL_m

        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        cos_theta = np.clip(dz / r, 0, 1)

        E = self.intensity * cos_theta / (r ** 2 + 1e-9)

        if self.spot_radius is not None:
            dist_xy = np.sqrt((X - self.x) ** 2 + (Y - self.y) ** 2)
            E[dist_xy > self.spot_radius] = 0  # обрубаем строго по кругу

        return E
