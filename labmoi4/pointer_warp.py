import ctypes
import shutil
import subprocess
import sys


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


class MouseWarp:
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


_warper_singleton = None


def mouse_warp():
    global _warper_singleton
    if sys.platform != "linux":
        return None
    if _warper_singleton is False:
        return None
    if _warper_singleton is None:
        try:
            w = MouseWarp()
        except OSError:
            _warper_singleton = False
            return None
        if not w.available():
            _warper_singleton = False
            return None
        _warper_singleton = w
    return _warper_singleton


def pointer_cursor_unlock(warper, wid):
    if warper is not None and wid is not None:
        warper.pointer_lock_deactivate(wid)
