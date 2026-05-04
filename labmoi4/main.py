import taichi_setup

taichi_setup.init_taichi()

import taichi as ti

import camera_control as cam
from camera_control import apply_mouse_look, look_direction, render_basis, walk_basis, walk_speed
from config import IMAGE_H, IMAGE_W, MAX_TRACE_DEPTH
import scene


def print_hotkeys():
    print(
        "\n=== labmoi4 hotkeys ===\n"
        "  3/4/5/6  samples per pixel (1/2/4/8)\n"
        "  [ ]      exposure darker / brighter\n"
        "  z x      DOF blur less / more\n"
        "  , .      FOV narrower / wider\n"
        "  - =      sky dimmer / brighter\n"
        "  9 0      max ray bounces − / +\n"
        "  k l      saturation − / +\n"
        "  h j      contrast − / +\n"
        "  b        bloom on/off\n"
        "  g        vignette on/off\n"
        "  m        animated lights on/off\n"
        "  r        reset accumulation\n"
        "  o        save PNG screenshot\n"
        "  Tab      show this help\n"
        "  WASD+QE  move; мышь — обзор\n"
        "========================\n"
    )


def main():
    scene.init_scene()
    scene.samples_wanted[None] = 2
    scene.exposure_mul[None] = 1.0
    scene.dof_radius[None] = 0.0015
    scene.bloom_on[None] = 1
    scene.vignette_on[None] = 1
    scene.scene_time[None] = 0.0
    scene.fov_scale[None] = 1.0
    scene.sky_mul[None] = 1.0
    scene.trace_depth_limit[None] = MAX_TRACE_DEPTH
    scene.saturation[None] = 1.0
    scene.contrast[None] = 1.0

    gui = ti.GUI("Path Tracing — labmoi4", (IMAGE_W, IMAGE_H))
    scene.sample_count[None] = 0
    print_hotkeys()

    while gui.running:
        if cam.control_cd > 0:
            cam.control_cd -= 1

        scene_changed = False

        if cam.motion_enabled:
            scene.scene_time[None] += 0.03
            scene.update_scene(scene.scene_time[None])
            scene_changed = True

        if gui.is_pressed("m") and cam.control_cd == 0:
            cam.motion_enabled = not cam.motion_enabled
            cam.control_cd = 15
            scene_changed = True

        for key, spp in (("3", 1), ("4", 2), ("5", 4), ("6", 8)):
            if gui.is_pressed(key):
                scene.samples_wanted[None] = spp
                scene_changed = True

        if gui.is_pressed("["):
            scene.exposure_mul[None] *= 0.9
        if gui.is_pressed("]"):
            scene.exposure_mul[None] *= 1.1

        if gui.is_pressed("z"):
            scene.dof_radius[None] *= 0.88
            scene_changed = True
        if gui.is_pressed("x"):
            scene.dof_radius[None] = min(scene.dof_radius[None] * 1.12, 0.06)
            scene_changed = True

        if gui.is_pressed(","):
            scene.fov_scale[None] *= 0.96
            scene.fov_scale[None] = max(scene.fov_scale[None], 0.2)
            scene_changed = True
        if gui.is_pressed("."):
            scene.fov_scale[None] *= 1.04
            scene.fov_scale[None] = min(scene.fov_scale[None], 3.5)
            scene_changed = True

        if gui.is_pressed("-"):
            scene.sky_mul[None] *= 0.92
            scene.sky_mul[None] = max(scene.sky_mul[None], 0.05)
            scene_changed = True
        if gui.is_pressed("="):
            scene.sky_mul[None] *= 1.08
            scene.sky_mul[None] = min(scene.sky_mul[None], 12.0)
            scene_changed = True

        if gui.is_pressed("9"):
            scene.trace_depth_limit[None] = max(scene.trace_depth_limit[None] - 2, 2)
            scene_changed = True
        if gui.is_pressed("0"):
            scene.trace_depth_limit[None] = min(scene.trace_depth_limit[None] + 2, MAX_TRACE_DEPTH)
            scene_changed = True

        if gui.is_pressed("k"):
            scene.saturation[None] *= 0.94
            scene.saturation[None] = max(scene.saturation[None], 0.0)
        if gui.is_pressed("l"):
            scene.saturation[None] *= 1.06
            scene.saturation[None] = min(scene.saturation[None], 2.5)

        if gui.is_pressed("h"):
            scene.contrast[None] *= 0.94
            scene.contrast[None] = max(scene.contrast[None], 0.35)
        if gui.is_pressed("j"):
            scene.contrast[None] *= 1.06
            scene.contrast[None] = min(scene.contrast[None], 2.5)

        if gui.is_pressed("b") and cam.control_cd == 0:
            scene.bloom_on[None] = 1 - scene.bloom_on[None]
            cam.control_cd = 8

        if gui.is_pressed("g") and cam.control_cd == 0:
            scene.vignette_on[None] = 1 - scene.vignette_on[None]
            cam.control_cd = 8

        if gui.is_pressed("r"):
            scene_changed = True

        if gui.is_pressed("Tab"):
            print_hotkeys()

        if gui.is_pressed("o"):
            ti.sync()
            fn = f"labmoi4_shot_{cam.shot_counter:04d}.png"
            ti.tools.imwrite(scene.image, fn)
            print(f"saved {fn}")
            cam.shot_counter += 1

        forward = look_direction()
        right_axis, up_axis = render_basis(forward)
        walk_forward, strafe_right = walk_basis(forward)

        if gui.is_pressed("w"):
            cam.eye += walk_speed * walk_forward
            scene_changed = True
        if gui.is_pressed("s"):
            cam.eye -= walk_speed * walk_forward
            scene_changed = True
        if gui.is_pressed("a"):
            cam.eye -= walk_speed * strafe_right
            scene_changed = True
        if gui.is_pressed("d"):
            cam.eye += walk_speed * strafe_right
            scene_changed = True
        if gui.is_pressed("q"):
            cam.eye += walk_speed * cam.WORLD_UP
            scene_changed = True
        if gui.is_pressed("e"):
            cam.eye -= walk_speed * cam.WORLD_UP
            scene_changed = True

        if apply_mouse_look(gui):
            scene_changed = True

        if scene_changed:
            scene.clear_accumulator()

        scene.camera_origin[None] = cam.eye
        scene.camera_forward[None] = forward
        scene.camera_right[None] = right_axis
        scene.camera_up[None] = up_axis

        scene.render()
        scene.tonemap()
        gui.set_image(scene.image)
        gui.show()
        scene.sample_count[None] += 1


if __name__ == "__main__":
    main()
