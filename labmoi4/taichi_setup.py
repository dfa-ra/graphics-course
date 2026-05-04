import taichi as ti


def init_taichi():
    ti.init(arch=ti.gpu)
    _patch_taichi_gui_motion_delta()


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
