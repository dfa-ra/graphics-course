import ctypes
import math
import shutil
import subprocess
import sys
import taichi as ti

ti.init(arch=ti.gpu)


def _patch_taichi_gui_motion_delta():
    from taichi.ui import gui as tg

    if getattr(tg.GUI, "_ti_motion_delta_patched", False):
        return

    def get_key_event_fixed(self):
        self.core.wait_key_event()
        e = tg.GUI.Event()
        event = self.core.get_key_event_head()
        e.type = event.type
        e.key = event.key
        e.pos = self.core.canvas_untransform(event.pos)
        e.pos = (e.pos[0], e.pos[1])
        e.modifier = []
        if e.key == tg.GUI.WHEEL:
            e.delta = event.delta
        elif e.type == tg.GUI.MOTION:
            e.delta = event.delta
        else:
            e.delta = (0, 0)
        for mod in ["Shift", "Alt", "Control"]:
            if self.is_pressed(mod):
                e.modifier.append(mod)
        if e.type == tg.GUI.PRESS:
            self.key_pressed.add(e.key)
        else:
            self.key_pressed.discard(e.key)
        self.core.pop_key_event_head()
        return e

    tg.GUI.get_key_event = get_key_event_fixed
    tg.GUI._ti_motion_delta_patched = True


_patch_taichi_gui_motion_delta()

IMAGE_W = 1920
IMAGE_H = 1080
ASPECT = float(IMAGE_W) / float(IMAGE_H)
MAX_SPP = 8
MAX_TRACE_DEPTH = 64
SPHERE_COUNT = 14

WORLD_UP = ti.Vector([0.0, 1.0, 0.0])

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

eye = ti.Vector([0.0, 1.0, 3.0])
yaw = 0.0
pitch = 0.0
walk_speed = 0.1
look_sensitivity = 0.003
_last_cursor = None
_pointer_locked = False
_skip_look_after_warp = False
_lock_target_window = None
_pending_window_capture = False
_warper_singleton = None

motion_enabled = False
_control_cd = 0
_shot_counter = 0


class _XColor(ctypes.Structure):
    _fields_ = [
        ("pixel", ctypes.c_ulong),
        ("red", ctypes.c_ushort),
        ("green", ctypes.c_ushort),
        ("blue", ctypes.c_ushort),
        ("flags", ctypes.c_char),
        ("pad", ctypes.c_char * 3),
    ]


class _X11Display:
    def __init__(self):
        self._lib = ctypes.CDLL("libX11.so.6")
        self._lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._lib.XOpenDisplay.restype = ctypes.c_void_p
        self._dpy = self._lib.XOpenDisplay(None)
        if not self._dpy:
            raise OSError("XOpenDisplay failed")
        self._lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self._lib.XDefaultRootWindow.restype = ctypes.c_ulong
        self._root = self._lib.XDefaultRootWindow(self._dpy)

        c_void_p = ctypes.c_void_p
        ulong = ctypes.c_ulong
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint

        self._lib.XGetInputFocus.argtypes = [c_void_p, ctypes.POINTER(ulong), ctypes.POINTER(c_int)]
        self._lib.XGetInputFocus.restype = ctypes.c_int

        self._lib.XGetGeometry.argtypes = [
            c_void_p,
            ulong,
            ctypes.POINTER(ulong),
            ctypes.POINTER(c_int),
            ctypes.POINTER(c_int),
            ctypes.POINTER(c_uint),
            ctypes.POINTER(c_uint),
            ctypes.POINTER(c_uint),
            ctypes.POINTER(c_uint),
        ]
        self._lib.XGetGeometry.restype = ctypes.c_int

        self._lib.XTranslateCoordinates.argtypes = [
            c_void_p,
            ulong,
            ulong,
            c_int,
            c_int,
            ctypes.POINTER(c_int),
            ctypes.POINTER(c_int),
            ctypes.POINTER(ulong),
        ]
        self._lib.XTranslateCoordinates.restype = ctypes.c_int

        self._lib.XWarpPointer.argtypes = [
            c_void_p,
            ulong,
            ulong,
            c_int,
            c_int,
            c_uint,
            c_uint,
            c_int,
            c_int,
        ]
        self._lib.XWarpPointer.restype = ctypes.c_int

        self._lib.XFlush.argtypes = [c_void_p]
        self._lib.XFlush.restype = ctypes.c_int

        self._lib.XCreatePixmap.argtypes = [c_void_p, ulong, c_uint, c_uint, c_uint]
        self._lib.XCreatePixmap.restype = ctypes.c_ulong

        self._lib.XCreatePixmapCursor.argtypes = [
            c_void_p,
            ulong,
            ulong,
            ctypes.POINTER(_XColor),
            ctypes.POINTER(_XColor),
            c_uint,
            c_uint,
        ]
        self._lib.XCreatePixmapCursor.restype = ctypes.c_ulong

        self._lib.XDefineCursor.argtypes = [c_void_p, ulong, ctypes.c_ulong]
        self._lib.XDefineCursor.restype = ctypes.c_int

        self._lib.XUndefineCursor.argtypes = [c_void_p, ulong]
        self._lib.XUndefineCursor.restype = ctypes.c_int

        self._lib.XGrabPointer.argtypes = [
            c_void_p,
            ulong,
            c_int,
            c_uint,
            c_int,
            c_int,
            ulong,
            ulong,
            ulong,
        ]
        self._lib.XGrabPointer.restype = ctypes.c_int

        self._lib.XUngrabPointer.argtypes = [c_void_p, ulong]
        self._lib.XUngrabPointer.restype = ctypes.c_int

        self._blank_pix = None
        self._blank_cur = None
        self._grab_active = False

    def _ensure_blank_cursor(self):
        if self._blank_cur is not None:
            return self._blank_cur
        pix = self._lib.XCreatePixmap(
            self._dpy,
            ctypes.c_ulong(self._root),
            ctypes.c_uint(1),
            ctypes.c_uint(1),
            ctypes.c_uint(1),
        )
        if not pix:
            return None
        fg = _XColor()
        bg = _XColor()
        cur = self._lib.XCreatePixmapCursor(
            self._dpy,
            ctypes.c_ulong(pix),
            ctypes.c_ulong(pix),
            ctypes.byref(fg),
            ctypes.byref(bg),
            ctypes.c_uint(0),
            ctypes.c_uint(0),
        )
        self._blank_pix = int(pix)
        self._blank_cur = int(cur)
        return self._blank_cur

    def current_focus_window(self):
        w = ctypes.c_ulong()
        rev = ctypes.c_int()
        self._lib.XGetInputFocus(self._dpy, ctypes.byref(w), ctypes.byref(rev))
        wid = int(w.value)
        if wid in (0, 1, 2):
            return None
        return wid

    def window_screen_center(self, wid):
        root_ret = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        if not self._lib.XGetGeometry(
            self._dpy,
            ctypes.c_ulong(wid),
            ctypes.byref(root_ret),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(depth),
        ):
            return None
        abs_x = ctypes.c_int()
        abs_y = ctypes.c_int()
        child = ctypes.c_ulong()
        if not self._lib.XTranslateCoordinates(
            self._dpy,
            ctypes.c_ulong(wid),
            ctypes.c_ulong(self._root),
            0,
            0,
            ctypes.byref(abs_x),
            ctypes.byref(abs_y),
            ctypes.byref(child),
        ):
            return None
        cx = int(abs_x.value) + int(width.value) // 2
        cy = int(abs_y.value) + int(height.value) // 2
        return cx, cy

    def warp_root(self, screen_x, screen_y):
        self._lib.XWarpPointer(
            self._dpy,
            ctypes.c_ulong(0),
            ctypes.c_ulong(self._root),
            0,
            0,
            0,
            0,
            int(screen_x),
            int(screen_y),
        )
        self._lib.XFlush(self._dpy)

    def warp_pointer_to_window_center(self, wid):
        root_ret = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        if not self._lib.XGetGeometry(
            self._dpy,
            ctypes.c_ulong(int(wid)),
            ctypes.byref(root_ret),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(depth),
        ):
            return False
        cx = max(int(width.value) // 2, 0)
        cy = max(int(height.value) // 2, 0)
        self._lib.XWarpPointer(
            self._dpy,
            ctypes.c_ulong(0),
            ctypes.c_ulong(int(wid)),
            0,
            0,
            0,
            0,
            cx,
            cy,
        )
        self._lib.XFlush(self._dpy)
        return True

    def grab_pointer(self, wid):
        if self._grab_active:
            return True
        event_mask = 0x40 | 0x100 | 0x200
        r = self._lib.XGrabPointer(
            self._dpy,
            ctypes.c_ulong(int(wid)),
            ctypes.c_int(1),
            ctypes.c_uint(event_mask),
            ctypes.c_int(1),
            ctypes.c_int(1),
            ctypes.c_ulong(int(wid)),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        if r == 0:
            self._grab_active = True
            self._lib.XFlush(self._dpy)
            return True
        return False

    def ungrab_pointer(self):
        if not self._grab_active:
            return
        self._lib.XUngrabPointer(self._dpy, ctypes.c_ulong(0))
        self._grab_active = False
        self._lib.XFlush(self._dpy)

    def hide_cursor_on_window(self, wid):
        cur = self._ensure_blank_cursor()
        if cur is None:
            return False
        self._lib.XDefineCursor(self._dpy, ctypes.c_ulong(int(wid)), ctypes.c_ulong(int(cur)))
        self._lib.XFlush(self._dpy)
        return True

    def show_cursor_on_window(self, wid):
        self._lib.XUndefineCursor(self._dpy, ctypes.c_ulong(int(wid)))
        self._lib.XFlush(self._dpy)


class _MouseWarp:
    def __init__(self):
        self._xdotool = shutil.which("xdotool")
        self._x11_cache = None

    def _ensure_x11(self):
        if self._x11_cache is False:
            return None
        if self._x11_cache is None:
            try:
                self._x11_cache = _X11Display()
            except OSError:
                self._x11_cache = False
        return self._x11_cache if self._x11_cache else None

    def available(self):
        return self._xdotool is not None or self._ensure_x11() is not None

    def active_window_id(self):
        if self._xdotool:
            try:
                out = subprocess.check_output(
                    [self._xdotool, "getactivewindow"],
                    text=True,
                    timeout=0.5,
                    stderr=subprocess.DEVNULL,
                ).strip()
                return int(out)
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass
        x11 = self._ensure_x11()
        return x11.current_focus_window() if x11 else None

    def window_center_root(self, wid):
        if self._xdotool:
            try:
                out = subprocess.check_output(
                    [self._xdotool, "getwindowgeometry", "--shell", str(wid)],
                    text=True,
                    timeout=0.5,
                    stderr=subprocess.DEVNULL,
                )
                kv = {}
                for line in out.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        kv[k.strip()] = int(v.strip())
                if all(k in kv for k in ("X", "Y", "WIDTH", "HEIGHT")):
                    return kv["X"] + kv["WIDTH"] // 2, kv["Y"] + kv["HEIGHT"] // 2
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError, KeyError):
                pass
        x11 = self._ensure_x11()
        return x11.window_screen_center(wid) if x11 else None

    def warp_to_window_center(self, wid):
        if wid is None:
            return False
        wid = int(wid)
        x11 = self._ensure_x11()
        if x11 is not None:
            if x11.warp_pointer_to_window_center(wid):
                return True
        c = self.window_center_root(wid)
        if c is None:
            return False
        rx, ry = int(c[0]), int(c[1])
        if self._xdotool:
            try:
                subprocess.run(
                    [self._xdotool, "mousemove", "--sync", str(rx), str(ry)],
                    timeout=0.5,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        if x11:
            x11.warp_root(rx, ry)
            return True
        return False

    def pointer_lock_activate(self, wid):
        if wid is None:
            return False
        wid = int(wid)
        x11 = self._ensure_x11()
        if x11 is None:
            return self.warp_to_window_center(wid)
        x11.warp_pointer_to_window_center(wid)
        hidden = x11.hide_cursor_on_window(wid)
        if not x11.grab_pointer(wid):
            if hidden:
                x11.show_cursor_on_window(wid)
            return False
        return True

    def pointer_lock_deactivate(self, wid):
        x11 = self._ensure_x11()
        if x11 is None:
            return
        x11.ungrab_pointer()
        if wid is not None:
            x11.show_cursor_on_window(int(wid))


def _mouse_warp():
    global _warper_singleton
    if sys.platform != "linux":
        return None
    if _warper_singleton is False:
        return None
    if _warper_singleton is None:
        try:
            w = _MouseWarp()
        except OSError:
            _warper_singleton = False
            return None
        if not w.available():
            _warper_singleton = False
            return None
        _warper_singleton = w
    return _warper_singleton


def _pointer_cursor_unlock(warper, wid):
    if warper is not None and wid is not None:
        warper.pointer_lock_deactivate(wid)


def look_direction():
    f = ti.Vector(
        [
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw),
        ]
    )
    return f.normalized()


def render_basis(forward):
    r = forward.cross(WORLD_UP)
    if r.norm() < 1e-5:
        r = ti.Vector([1.0, 0.0, 0.0])
    else:
        r = r.normalized()
    u = r.cross(forward).normalized()
    return r, u


def walk_basis(forward):
    flat = ti.Vector([forward.x, 0.0, forward.z])
    if flat.norm() < 1e-5:
        flat = ti.Vector([0.0, 0.0, 1.0])
    else:
        flat = flat.normalized()
    strafe = flat.cross(WORLD_UP).normalized()
    return flat, strafe


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

    for depth in range(MAX_TRACE_DEPTH):
        hit_id, t, normal = hit_spheres(origin, direction)
        if hit_id == -1:
            color += throughput * sky_color(direction)
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
                ray_dir = (
                    camera_forward[None]
                    + (u - 0.5) * 2.0 * ASPECT * camera_right[None]
                    + (v - 0.5) * 2.0 * camera_up[None]
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


def print_hotkeys():
    print(
        "\n=== labmoi4 hotkeys ===\n"
        "  3/4/5/6  samples per pixel (1/2/4/8)\n"
        "  [ ]      exposure darker / brighter\n"
        "  z x      DOF blur less / more\n"
        "  b        bloom on/off\n"
        "  g        vignette on/off\n"
        "  m        animated lights on/off\n"
        "  r        reset accumulation\n"
        "  o        save PNG screenshot\n"
        "  Tab      show this help\n"
        "  WASD+QE  move; LMB — FPS-захват (курсор скрыт, центр окна)\n"
        "========================\n"
    )


def apply_mouse_look(gui):
    global yaw, pitch, _last_cursor
    global _pointer_locked, _skip_look_after_warp, _lock_target_window
    global _pending_window_capture

    lim = math.pi / 2 - 0.01
    warper = _mouse_warp()

    motion_dx, motion_dy = 0.0, 0.0
    while gui.get_event():
        ev = gui.event
        if ev.type == ti.GUI.MOTION:
            motion_dx += float(ev.delta[0])
            motion_dy += float(ev.delta[1])
        elif ev.type == ti.GUI.PRESS:
            if ev.key == ti.GUI.LMB:
                if _pointer_locked:
                    lw = _lock_target_window
                    _pointer_cursor_unlock(warper, lw)
                    _pointer_locked = False
                    _lock_target_window = None
                    _skip_look_after_warp = False
                    _pending_window_capture = False
                    _last_cursor = None
                else:
                    _pointer_locked = True
                    _lock_target_window = None
                    _skip_look_after_warp = True
                    _last_cursor = None
                    _pending_window_capture = warper is not None
            elif ev.key == ti.GUI.ESCAPE and _pointer_locked:
                lw = _lock_target_window
                _pointer_cursor_unlock(warper, lw)
                _pointer_locked = False
                _lock_target_window = None
                _skip_look_after_warp = False
                _pending_window_capture = False
                _last_cursor = None

    if _pending_window_capture:
        if _pointer_locked and warper is not None:
            wid = warper.active_window_id()
            if wid is not None and wid not in (0, 1, 2):
                _lock_target_window = wid
        _pending_window_capture = False

    changed = False

    if _pointer_locked:
        if (
            warper is not None
            and _lock_target_window is not None
            and warper.active_window_id() != _lock_target_window
        ):
            lw = _lock_target_window
            _pointer_cursor_unlock(warper, lw)
            _pointer_locked = False
            _lock_target_window = None
            _skip_look_after_warp = False
            _pending_window_capture = False
            _last_cursor = None
        else:
            if _skip_look_after_warp:
                ok = False
                if warper is not None and _lock_target_window is not None:
                    ok = warper.pointer_lock_activate(_lock_target_window)
                if not ok:
                    _pointer_locked = False
                    _lock_target_window = None
                _skip_look_after_warp = False
                _last_cursor = gui.get_cursor_pos()
                return changed

            if motion_dx != 0.0 or motion_dy != 0.0:
                yaw += motion_dx * look_sensitivity
                pitch -= motion_dy * look_sensitivity
                pitch = max(-lim, min(lim, pitch))
                changed = True

            cx, cy = gui.get_cursor_pos()
            if not changed and _last_cursor is not None:
                dcx, dcy = cx - _last_cursor[0], cy - _last_cursor[1]
                if dcx != 0.0 or dcy != 0.0:
                    yaw += dcx * IMAGE_W * look_sensitivity
                    pitch += dcy * IMAGE_H * look_sensitivity
                    pitch = max(-lim, min(lim, pitch))
                    changed = True

            if warper is not None and _lock_target_window is not None:
                warper.warp_to_window_center(_lock_target_window)
            _last_cursor = gui.get_cursor_pos()
            return changed

    cx, cy = gui.get_cursor_pos()
    if _last_cursor is not None:
        dcx, dcy = cx - _last_cursor[0], cy - _last_cursor[1]
        if dcx != 0.0 or dcy != 0.0:
            yaw += dcx * IMAGE_W * look_sensitivity
            pitch += dcy * IMAGE_H * look_sensitivity
            pitch = max(-lim, min(lim, pitch))
            changed = True
    _last_cursor = (cx, cy)
    return changed


def main():
    global eye, yaw, pitch, motion_enabled, _control_cd, _shot_counter

    init_scene()
    samples_wanted[None] = 2
    exposure_mul[None] = 1.0
    dof_radius[None] = 0.0015
    bloom_on[None] = 1
    vignette_on[None] = 1
    scene_time[None] = 0.0

    gui = ti.GUI("Path Tracing — labmoi4", (IMAGE_W, IMAGE_H))
    sample_count[None] = 0
    print_hotkeys()

    while gui.running:
        if _control_cd > 0:
            _control_cd -= 1

        scene_changed = False

        if motion_enabled:
            scene_time[None] += 0.03
            update_scene(scene_time[None])
            scene_changed = True

        if gui.is_pressed("m") and _control_cd == 0:
            motion_enabled = not motion_enabled
            _control_cd = 15
            scene_changed = True

        for key, spp in (("3", 1), ("4", 2), ("5", 4), ("6", 8)):
            if gui.is_pressed(key):
                samples_wanted[None] = spp
                scene_changed = True

        if gui.is_pressed("["):
            exposure_mul[None] *= 0.9
        if gui.is_pressed("]"):
            exposure_mul[None] *= 1.1

        if gui.is_pressed("z"):
            dof_radius[None] *= 0.88
            scene_changed = True
        if gui.is_pressed("x"):
            dof_radius[None] = min(dof_radius[None] * 1.12, 0.06)
            scene_changed = True

        if gui.is_pressed("b") and _control_cd == 0:
            bloom_on[None] = 1 - bloom_on[None]
            _control_cd = 8

        if gui.is_pressed("g") and _control_cd == 0:
            vignette_on[None] = 1 - vignette_on[None]
            _control_cd = 8

        if gui.is_pressed("r"):
            scene_changed = True

        if gui.is_pressed("Tab"):
            print_hotkeys()

        if gui.is_pressed("o"):
            ti.sync()
            fn = f"labmoi4_shot_{_shot_counter:04d}.png"
            ti.tools.imwrite(image, fn)
            print(f"saved {fn}")
            _shot_counter += 1

        forward = look_direction()
        right_axis, up_axis = render_basis(forward)
        walk_forward, strafe_right = walk_basis(forward)

        if gui.is_pressed("w"):
            eye += walk_speed * walk_forward
            scene_changed = True
        if gui.is_pressed("s"):
            eye -= walk_speed * walk_forward
            scene_changed = True
        if gui.is_pressed("a"):
            eye -= walk_speed * strafe_right
            scene_changed = True
        if gui.is_pressed("d"):
            eye += walk_speed * strafe_right
            scene_changed = True
        if gui.is_pressed("q"):
            eye += walk_speed * WORLD_UP
            scene_changed = True
        if gui.is_pressed("e"):
            eye -= walk_speed * WORLD_UP
            scene_changed = True

        if apply_mouse_look(gui):
            scene_changed = True

        if scene_changed:
            clear_accumulator()

        camera_origin[None] = eye
        camera_forward[None] = forward
        camera_right[None] = right_axis
        camera_up[None] = up_axis

        render()
        tonemap()
        gui.set_image(image)
        gui.show()
        warper = _mouse_warp()
        if _pointer_locked and _lock_target_window is not None and warper is not None:
            warper.warp_to_window_center(_lock_target_window)
        sample_count[None] += 1


if __name__ == "__main__":
    main()
