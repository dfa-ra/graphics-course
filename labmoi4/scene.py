import taichi as ti

from config import (
    ASPECT,
    IMAGE_H,
    IMAGE_W,
    MAX_SPP,
    MAX_TRACE_DEPTH,
    SPHERE_COUNT,
)

image = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
sample_count = ti.field(dtype=ti.i32, shape=())

camera_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
camera_forward = ti.Vector.field(3, dtype=ti.f32, shape=())
camera_right = ti.Vector.field(3, dtype=ti.f32, shape=())
camera_up = ti.Vector.field(3, dtype=ti.f32, shape=())

sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=SPHERE_COUNT)
sphere_radius = ti.field(dtype=ti.f32, shape=SPHERE_COUNT)
sphere_albedo = ti.Vector.field(3, dtype=ti.f32, shape=SPHERE_COUNT)
sphere_emission = ti.Vector.field(3, dtype=ti.f32, shape=SPHERE_COUNT)
sphere_metallic = ti.field(dtype=ti.f32, shape=SPHERE_COUNT)

samples_wanted = ti.field(dtype=ti.i32, shape=())
exposure_mul = ti.field(dtype=ti.f32, shape=())
dof_radius = ti.field(dtype=ti.f32, shape=())
bloom_on = ti.field(dtype=ti.i32, shape=())
vignette_on = ti.field(dtype=ti.i32, shape=())
scene_time = ti.field(dtype=ti.f32, shape=())
# FOV: множитель к смещению луча в плоскости камеры (меньше — уже кадр / «теле»).
fov_scale = ti.field(dtype=ti.f32, shape=())
sky_mul = ti.field(dtype=ti.f32, shape=())
trace_depth_limit = ti.field(dtype=ti.i32, shape=())
saturation = ti.field(dtype=ti.f32, shape=())
contrast = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def init_scene():
    sphere_center[0] = ti.Vector([0.0, -1001.0, 0.0])
    sphere_radius[0] = 1000.0
    sphere_albedo[0] = ti.Vector([0.22, 0.24, 0.32])
    sphere_emission[0] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[0] = 0.0

    sphere_center[1] = ti.Vector([0.0, 0.35, -3.2])
    sphere_radius[1] = 0.85
    sphere_albedo[1] = ti.Vector([0.92, 0.2, 0.35])
    sphere_emission[1] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[1] = 0.0

    sphere_center[2] = ti.Vector([2.1, 0.4, -4.0])
    sphere_radius[2] = 0.9
    sphere_albedo[2] = ti.Vector([0.95, 0.95, 0.98])
    sphere_emission[2] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[2] = 1.0

    sphere_center[3] = ti.Vector([0.0, 4.8, -1.9])
    sphere_radius[3] = 0.55
    sphere_albedo[3] = ti.Vector([1.0, 1.0, 1.0])
    sphere_emission[3] = ti.Vector([14.0, 14.0, 16.0])
    sphere_metallic[3] = 0.0

    sphere_center[4] = ti.Vector([-2.2, 0.55, -3.6])
    sphere_radius[4] = 0.55
    sphere_albedo[4] = ti.Vector([0.15, 0.85, 0.55])
    sphere_emission[4] = ti.Vector([0.0, 2.5, 1.2])
    sphere_metallic[4] = 0.15

    sphere_center[5] = ti.Vector([-1.0, 0.2, -2.4])
    sphere_radius[5] = 0.22
    sphere_albedo[5] = ti.Vector([0.95, 0.75, 0.2])
    sphere_emission[5] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[5] = 0.9

    sphere_center[6] = ti.Vector([1.2, 0.18, -2.1])
    sphere_radius[6] = 0.18
    sphere_albedo[6] = ti.Vector([0.2, 0.45, 1.0])
    sphere_emission[6] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[6] = 1.0

    sphere_center[7] = ti.Vector([3.0, 0.25, -5.5])
    sphere_radius[7] = 0.28
    sphere_albedo[7] = ti.Vector([1.0, 0.35, 0.9])
    sphere_emission[7] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[7] = 0.85

    sphere_center[8] = ti.Vector([-2.4, 1.2, -6.0])
    sphere_radius[8] = 0.4
    sphere_albedo[8] = ti.Vector([0.05, 0.05, 0.05])
    sphere_emission[8] = ti.Vector([2.0, 4.0, 12.0])
    sphere_metallic[8] = 0.0

    sphere_center[9] = ti.Vector([0.8, 0.12, -1.9])
    sphere_radius[9] = 0.12
    sphere_albedo[9] = ti.Vector([0.9, 0.9, 0.9])
    sphere_emission[9] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[9] = 1.0

    sphere_center[10] = ti.Vector([-0.6, 0.1, -1.7])
    sphere_radius[10] = 0.1
    sphere_albedo[10] = ti.Vector([0.4, 1.0, 0.5])
    sphere_emission[10] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[10] = 0.0

    sphere_center[11] = ti.Vector([2.6, 0.08, -2.6])
    sphere_radius[11] = 0.08
    sphere_albedo[11] = ti.Vector([1.0, 0.4, 0.1])
    sphere_emission[11] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[11] = 1.0

    sphere_center[12] = ti.Vector([-1.4, 0.45, -4.8])
    sphere_radius[12] = 0.45
    sphere_albedo[12] = ti.Vector([0.02, 0.02, 0.02])
    sphere_emission[12] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[12] = 1.0

    sphere_center[13] = ti.Vector([4.2, 0.6, -7.0])
    sphere_radius[13] = 0.6
    sphere_albedo[13] = ti.Vector([0.85, 0.9, 1.0])
    sphere_emission[13] = ti.Vector([0.0, 0.0, 0.0])
    sphere_metallic[13] = 0.35


@ti.kernel
def update_scene(t: ti.f32):
    sphere_center[3] = ti.Vector(
        [
            ti.sin(t * 0.7) * 1.8,
            4.8 + ti.sin(t * 1.3) * 0.35,
            -3.0 + ti.cos(t * 0.55) * 1.1,
        ]
    )
    sphere_center[8] = ti.Vector(
        [
            -3.0 + ti.cos(t * 0.4) * 0.6,
            1.2 + ti.sin(t * 0.9) * 0.25,
            -6.0 + ti.sin(t * 0.35) * 0.8,
        ]
    )
    sphere_emission[4] = ti.Vector(
        [
            0.0,
            1.8 + ti.sin(t * 2.0) * 0.8,
            0.8 + ti.cos(t * 1.7) * 0.5,
        ]
    )


@ti.func
def random_in_unit_sphere():
    p = ti.Vector([0.0, 0.0, 0.0])
    for _ in range(10):
        candidate = ti.Vector(
            [
                ti.random() * 2 - 1,
                ti.random() * 2 - 1,
                ti.random() * 2 - 1,
            ]
        )
        if candidate.norm() < 1:
            p = candidate
    return p


@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n


@ti.func
def sky_color(d):
    t = 0.5 * (ti.min(ti.max(d.y, -1.0), 1.0) + 1.0)
    bot = ti.Vector([0.03, 0.035, 0.08])
    top = ti.Vector([0.1, 0.35, 0.72])
    band = ti.Vector([0.55, 0.12, 0.45])
    a = ti.atan2(d.z, d.x)
    streak = ti.pow(ti.abs(ti.sin(a * 3.0 + d.y * 4.0)) * 0.5 + 0.5, 8.0)
    c = (1.0 - t) * bot + t * top + streak * band * 0.22
    return ti.min(c, ti.Vector([2.5, 2.5, 2.5]))


@ti.func
def aces_approx(c):
    a = 2.51
    b = 0.03
    cc = 2.43
    d = 0.59
    e = 0.14
    r = ti.max((c[0] * (a * c[0] + b)) / (c[0] * (cc * c[0] + d) + e), 0.0)
    g = ti.max((c[1] * (a * c[1] + b)) / (c[1] * (cc * c[1] + d) + e), 0.0)
    bl = ti.max((c[2] * (a * c[2] + b)) / (c[2] * (cc * c[2] + d) + e), 0.0)
    return ti.Vector([r, g, bl])


@ti.func
def hit_spheres(origin, direction):
    closest_t = 1e8
    hit_index = -1
    hit_normal = ti.Vector([0.0, 0.0, 0.0])

    for i in range(SPHERE_COUNT):
        oc = origin - sphere_center[i]
        a = direction.dot(direction)
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - sphere_radius[i] * sphere_radius[i]
        disc = b * b - 4 * a * c
        if disc > 0:
            t = (-b - ti.sqrt(disc)) / (2.0 * a)
            if 0.001 < t < closest_t:
                closest_t = t
                hit_index = i
                hit_point = origin + t * direction
                hit_normal = (hit_point - sphere_center[i]).normalized()

    return hit_index, closest_t, hit_normal


@ti.func
def trace(origin, direction):
    color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    lim = trace_depth_limit[None]
    if lim < 2:
        lim = 2
    if lim > MAX_TRACE_DEPTH:
        lim = MAX_TRACE_DEPTH

    for depth in range(MAX_TRACE_DEPTH):
        if depth >= lim:
            break
        hit_id, t, normal = hit_spheres(origin, direction)
        if hit_id == -1:
            color += throughput * sky_color(direction) * sky_mul[None]
            break

        hit_point = origin + t * direction
        color += throughput * sphere_emission[hit_id]

        if depth > 2:
            survival = max(throughput.max(), 0.1)
            if ti.random() > survival:
                break
            throughput /= survival

        if sphere_metallic[hit_id] > 0.5:
            direction = reflect(direction, normal)
        else:
            direction = (normal + random_in_unit_sphere()).normalized()

        throughput *= sphere_albedo[hit_id]
        origin = hit_point + normal * 0.001

    return color


@ti.kernel
def render():
    for x, y in image:
        pixel = ti.Vector([0.0, 0.0, 0.0])
        nspp = samples_wanted[None]
        if nspp > MAX_SPP:
            nspp = MAX_SPP
        rad = dof_radius[None]
        for si in range(MAX_SPP):
            if si < nspp:
                u = (x + ti.random()) / IMAGE_W
                v = (y + ti.random()) / IMAGE_H
                ox = camera_origin[None]
                if rad > 1e-5:
                    ox = (
                        ox
                        + camera_right[None] * (ti.random() * 2.0 - 1.0) * rad
                        + camera_up[None] * (ti.random() * 2.0 - 1.0) * rad
                    )
                fs = fov_scale[None]
                fs = ti.max(0.12, ti.min(fs, 4.0))
                ray_dir = (
                    camera_forward[None]
                    + (u - 0.5) * 2.0 * ASPECT * camera_right[None] * fs
                    + (v - 0.5) * 2.0 * camera_up[None] * fs
                ).normalized()
                pixel += trace(ox, ray_dir)
        accumulator[x, y] += pixel
        n = sample_count[None] + 1
        image[x, y] = accumulator[x, y] / n


@ti.kernel
def tonemap():
    for x, y in image:
        c = image[x, y] * exposure_mul[None]
        c = aces_approx(c)
        if bloom_on[None] != 0:
            br = ti.max(c[0] - 0.52, 0.0)
            bg = ti.max(c[1] - 0.52, 0.0)
            bb = ti.max(c[2] - 0.55, 0.0)
            c = c + ti.Vector([br, bg, bb]) * 1.15
        if vignette_on[None] != 0:
            u = (ti.cast(x, ti.f32) + 0.5) / IMAGE_W - 0.5
            v = (ti.cast(y, ti.f32) + 0.5) / IMAGE_H - 0.5
            r2 = u * u + v * v
            c = c * (1.0 - 0.48 * r2)
        s = saturation[None]
        s = ti.max(s, 0.0)
        gray = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        gv = ti.Vector([gray, gray, gray])
        c = gv + s * (c - gv)
        c = ti.max(c, ti.Vector([0.0, 0.0, 0.0]))
        k = contrast[None]
        k = ti.max(k, 0.05)
        c = (c - ti.Vector([0.5, 0.5, 0.5])) * k + ti.Vector([0.5, 0.5, 0.5])
        c = ti.max(c, ti.Vector([0.0, 0.0, 0.0]))
        c = ti.Vector(
            [
                ti.pow(ti.min(c[0], 4.0), 1.0 / 2.2),
                ti.pow(ti.min(c[1], 4.0), 1.0 / 2.2),
                ti.pow(ti.min(c[2], 4.0), 1.0 / 2.2),
            ]
        )
        image[x, y] = ti.min(c, ti.Vector([1.0, 1.0, 1.0]))


def clear_accumulator():
    accumulator.fill(0)
    sample_count[None] = 0
