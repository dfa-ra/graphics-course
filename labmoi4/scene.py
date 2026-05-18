import taichi as ti

from config import (
    ASPECT,
    BILATERAL_RADIUS,
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

# Гиды для билатерального фильтра (первое пересечение луча со сценой).
OBJ_HIST_BINS = SPHERE_COUNT + 1

geom_depth_sum = ti.field(dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
geom_normal_sum = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
geom_hit_samples = ti.field(dtype=ti.i32, shape=(IMAGE_W, IMAGE_H))
obj_hist = ti.field(dtype=ti.i32, shape=(IMAGE_W, IMAGE_H, OBJ_HIST_BINS))

geom_depth_guide = ti.field(dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
geom_normal_guide = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
dominant_obj = ti.field(dtype=ti.i32, shape=(IMAGE_W, IMAGE_H))

filtered_hdr = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_W, IMAGE_H))
bilateral_on = ti.field(dtype=ti.i32, shape=())

# Суммы яркости по доминирующему объекту (−1..SPHERE_COUNT−1 → индекс 0..SPHERE_COUNT).
lum_in_per_obj = ti.field(dtype=ti.f32, shape=(SPHERE_COUNT + 1,))
lum_out_per_obj = ti.field(dtype=ti.f32, shape=(SPHERE_COUNT + 1,))


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
def trace_with_primary(origin, direction):
    """Полная энергия + геометрия первого пересечения (для билатерального фильтра)."""
    color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    prim_depth = 0.0
    prim_n = ti.Vector([0.0, 0.0, 0.0])
    prim_id = -1

    lim = trace_depth_limit[None]
    if lim < 2:
        lim = 2
    if lim > MAX_TRACE_DEPTH:
        lim = MAX_TRACE_DEPTH

    o = origin
    d = direction

    for depth in range(MAX_TRACE_DEPTH):
        if depth >= lim:
            break
        hit_id, t, normal = hit_spheres(o, d)
        if hit_id == -1:
            color += throughput * sky_color(d) * sky_mul[None]
            break

        hit_point = o + t * d
        color += throughput * sphere_emission[hit_id]

        if depth == 0:
            prim_depth = t
            prim_n = normal
            prim_id = hit_id

        if depth > 2:
            survival = max(throughput.max(), 0.1)
            if ti.random() > survival:
                break
            throughput /= survival

        m = sphere_metallic[hit_id]
        m = ti.min(ti.max(m, 0.0), 1.0)
        if ti.random() < m:
            d = reflect(d, normal)
        else:
            d = (normal + random_in_unit_sphere()).normalized()

        throughput *= sphere_albedo[hit_id]
        o = hit_point + normal * 0.001

    return color, prim_depth, prim_n, prim_id

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
                col, pz, pn, pid = trace_with_primary(ox, ray_dir)
                pixel += col
                if pid >= 0:
                    geom_depth_sum[x, y] += pz
                    geom_normal_sum[x, y] += pn
                    geom_hit_samples[x, y] += 1
                    ti.atomic_add(obj_hist[x, y, pid + 1], 1)
                else:
                    ti.atomic_add(obj_hist[x, y, 0], 1)
        accumulator[x, y] += pixel
        n = sample_count[None] + 1
        image[x, y] = accumulator[x, y] / n


@ti.func
def luminance(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


@ti.kernel
def resolve_geom_guides_and_dominant():
    """Средняя глубина/нормаль по сэмплам и доминирующий id объекта (гистограмма)."""
    for x, y in geom_depth_guide:
        hc = geom_hit_samples[x, y]
        zg = 0.0
        ng = ti.Vector([0.0, 0.0, 0.0])
        if hc > 0:
            zg = geom_depth_sum[x, y] / ti.cast(hc, ti.f32)
            ng = (geom_normal_sum[x, y] / ti.cast(hc, ti.f32)).normalized()
        geom_depth_guide[x, y] = zg
        geom_normal_guide[x, y] = ng

        best = -1
        dom_bin = 0
        for b in ti.static(range(OBJ_HIST_BINS)):
            v = obj_hist[x, y, b]
            if v > best:
                best = v
                dom_bin = b
        # bin 0 — небо/промах; bin k+1 — сфера k
        dominant_obj[x, y] = dom_bin - 1


@ti.kernel
def bilateral_filter_hdr():
    """Билатераль по яркости + веса по глубине, нормали и индексу объекта (сохранение границ)."""
    half = BILATERAL_RADIUS
    sig_s = 1.25
    sig_r = 0.07
    sig_z = 0.55
    sig_n = 0.35
    different_obj_scale = 0.04

    for x, y in filtered_hdr:
        ip = image[x, y]
        zp = geom_depth_guide[x, y]
        np_ = geom_normal_guide[x, y]
        id_p = dominant_obj[x, y]

        wp_sum = 0.0
        acc = ti.Vector([0.0, 0.0, 0.0])
        lp = luminance(ip)

        for j in range(-half, half + 1):
            for i in range(-half, half + 1):
                xi = ti.max(0, ti.min(IMAGE_W - 1, x + i))
                yj = ti.max(0, ti.min(IMAGE_H - 1, y + j))
                iq = image[xi, yj]

                dz = zp - geom_depth_guide[xi, yj]
                nd = ti.max(np_.dot(geom_normal_guide[xi, yj]), 0.0)

                gz = ti.exp(-(dz * dz) / (2.0 * sig_z * sig_z + 1e-6))
                zn = zp * geom_depth_guide[xi, yj]
                gz = ti.select(zn < 1e-8, 1.0, gz)

                gn = ti.exp(-((1.0 - nd) * (1.0 - nd)) / (2.0 * sig_n * sig_n + 1e-6))
                if geom_hit_samples[x, y] <= 0 or geom_hit_samples[xi, yj] <= 0:
                    gn = 1.0

                g_id = 1.0
                if id_p != dominant_obj[xi, yj]:
                    g_id = different_obj_scale

                ds = ti.cast(i * i + j * j, ti.f32)
                gs = ti.exp(-ds / (2.0 * sig_s * sig_s + 1e-6))

                lq = luminance(iq)
                dr = lp - lq
                gr = ti.exp(-(dr * dr) / (2.0 * sig_r * sig_r + 1e-6))

                w = gs * gr * gz * gn * g_id
                acc += iq * w
                wp_sum += w

        if wp_sum > 1e-8:
            filtered_hdr[x, y] = acc / wp_sum
        else:
            filtered_hdr[x, y] = ip


@ti.kernel
def clear_obj_luma_sums():
    for i in range(SPHERE_COUNT + 1):
        lum_in_per_obj[i] = 0.0
        lum_out_per_obj[i] = 0.0


@ti.kernel
def accumulate_obj_luma_sums():
    for x, y in image:
        idx = dominant_obj[x, y] + 1
        li = luminance(image[x, y])
        lo = luminance(filtered_hdr[x, y])
        ti.atomic_add(lum_in_per_obj[idx], li)
        ti.atomic_add(lum_out_per_obj[idx], lo)


@ti.kernel
def apply_obj_luminance_scales():
    for x, y in filtered_hdr:
        idx = dominant_obj[x, y] + 1
        denom = lum_out_per_obj[idx]
        sc = 1.0
        if denom > 1e-6:
            sc = lum_in_per_obj[idx] / denom
        filtered_hdr[x, y] = filtered_hdr[x, y] * sc


def preserve_energy_per_object():
    """Нормировка по объекту: сумма яркости после фильтра = до (как в методичке)."""
    clear_obj_luma_sums()
    accumulate_obj_luma_sums()
    apply_obj_luminance_scales()


@ti.kernel
def tonemap():
    for x, y in image:
        hdr = ti.Vector([0.0, 0.0, 0.0])
        if bilateral_on[None] != 0:
            hdr = filtered_hdr[x, y]
        else:
            hdr = image[x, y]
        c = hdr * exposure_mul[None]
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
    geom_depth_sum.fill(0)
    geom_normal_sum.fill(0)
    geom_hit_samples.fill(0)
    obj_hist.fill(0)
