import math

import taichi as ti

from config import IMAGE_H, IMAGE_W

WORLD_UP = ti.Vector([0.0, 1.0, 0.0])

eye = ti.Vector([0.0, 1.0, 3.0])
yaw = 0.0
pitch = 0.0
walk_speed = 0.1
look_sensitivity = 0.003
last_cursor = None
motion_enabled = False
control_cd = 0
shot_counter = 0


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


def apply_mouse_look(gui):
    global yaw, pitch, last_cursor

    lim = math.pi / 2 - 0.01

    motion_dx, motion_dy = 0.0, 0.0
    while gui.get_event():
        ev = gui.event
        if ev.type == ti.GUI.MOTION:
            motion_dx += float(ev.delta[0])
            motion_dy += float(ev.delta[1])

    changed = False
    cx, cy = gui.get_cursor_pos()

    if motion_dx != 0.0 or motion_dy != 0.0:
        yaw += motion_dx * look_sensitivity
        pitch -= motion_dy * look_sensitivity
        pitch = max(-lim, min(lim, pitch))
        changed = True
    elif last_cursor is not None:
        dcx, dcy = cx - last_cursor[0], cy - last_cursor[1]
        if dcx != 0.0 or dcy != 0.0:
            yaw += dcx * IMAGE_W * look_sensitivity
            pitch += dcy * IMAGE_H * look_sensitivity
            pitch = max(-lim, min(lim, pitch))
            changed = True

    last_cursor = (cx, cy)
    return changed
