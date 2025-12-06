import re
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import taichi as ti
from PIL import Image, ImageTk

ti.init(arch=ti.cpu)

MAX_SPHERES = 32
MAX_LIGHTS = 8

SCREEN_Z = 0.0
W_MM = 500.0
H_MM = 300.0

sphere_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPHERES)
sphere_rad = ti.field(dtype=ti.f32, shape=MAX_SPHERES)
sphere_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPHERES)
sphere_active = ti.field(dtype=ti.i32, shape=())  # count

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_I0 = ti.field(dtype=ti.f32, shape=MAX_LIGHTS)
light_col = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LIGHTS)
light_active = ti.field(dtype=ti.i32, shape=())  # count

kd_field = ti.field(dtype=ti.f32, shape=())
ks_field = ti.field(dtype=ti.f32, shape=())
shininess_field = ti.field(dtype=ti.f32, shape=())
shadows_field = ti.field(dtype=ti.i32, shape=())  # 0/1

cam_x_field = ti.field(dtype=ti.f32, shape=())
cam_y_field = ti.field(dtype=ti.f32, shape=())
cam_z_field = ti.field(dtype=ti.f32, shape=())

img_field = None
IMG_W = 0
IMG_H = 0


def parse_lights(text):
    out = []
    parts = text.split(';')
    for p in parts:
        p = p.strip()
        if not p:
            continue
        nums = re.findall(r'-?\d+\.?\d*', p)
        if len(nums) == 4:
            x, y, z, I0 = map(float, nums)
            col = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif len(nums) == 7:
            x, y, z, I0, r, g, b = map(float, nums)
            col = np.array([r, g, b], dtype=np.float32)
        else:
            continue
        out.append({"pos": np.array([x, y, z], dtype=np.float32), "I0": float(I0), "col": col})
    return out


def clamp(n, a, b):
    return max(a, min(b, n))


@ti.func
def normalize(v):
    n = ti.sqrt(v.dot(v))
    return v / n if n > 1e-8 else ti.Vector([0.0, 0.0, 0.0])


@ti.func
def intersect_sphere(ray_o, ray_d, idx):
    C = sphere_pos[idx]
    R = sphere_rad[idx]

    oc = ray_o - C
    a = ray_d.dot(ray_d)
    b = 2.0 * ray_d.dot(oc)
    c = oc.dot(oc) - R * R
    disc = b * b - 4 * a * c

    t = -1.0  # default (no hit)

    if disc >= 0.0:
        sqrt_d = ti.sqrt(disc)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        best = 1e9
        if t1 > 1e-6 and t1 < best:
            best = t1
        if t2 > 1e-6 and t2 < best:
            best = t2

        if best < 1e8:
            t = best

    return t


@ti.kernel
def render_kernel(width: int, height: int, w_mm: ti.f32, h_mm: ti.f32, screen_z: ti.f32, out: ti.types.ndarray()):
    kd = kd_field[None]
    ks = ks_field[None]
    shininess = shininess_field[None]
    shadows_on = shadows_field[None] == 1

    cam = ti.Vector([cam_x_field[None], cam_y_field[None], cam_z_field[None]])

    num_spheres = sphere_active[None]
    num_lights = light_active[None]

    center = ti.Vector([0.0, 0.0, 0.0])
    for s in range(num_spheres):
        center += sphere_pos[s]
    if num_spheres > 0:
        center /= num_spheres
    else:
        center = ti.Vector([0.0, 0.0, 0.0])

    forward = normalize(center - cam)

    world_up = ti.Vector([0.0, 1.0, 0.0])
    if abs(forward.dot(world_up)) > 0.999:
        world_up = ti.Vector([0.0, 0.0, 1.0])

    right = normalize(forward.cross(world_up))
    up = normalize(right.cross(forward))

    # ---------- Pixel size ----------
    half_w = w_mm / 2.0
    half_h = h_mm / 2.0

    for j in range(height):
        for i in range(width):
            # ---------- Ray through pixel ----------
            u = (i + 0.5) / width - 0.5
            v = 0.5 - (j + 0.5) / height
            dir_norm = normalize(forward + u * 2.0 * half_w / width * right + v * 2.0 * half_h / height * up)

            # --- Find nearest sphere ---
            nearest_t = 1e9
            nearest_idx = -1
            for s in range(num_spheres):
                t = intersect_sphere(cam, dir_norm, s)
                if t > 0.0 and t < nearest_t:
                    nearest_t = t
                    nearest_idx = s

            if nearest_idx == -1:
                out[j, i, 0] = 0.0
                out[j, i, 1] = 0.0
                out[j, i, 2] = 0.0
                continue

            # --- Shading ---
            P = cam + nearest_t * dir_norm
            C = sphere_pos[nearest_idx]
            N = normalize(P - C)
            V = normalize(cam - P)
            if N.dot(V) <= 0.0:
                out[j, i, 0] = 0.0
                out[j, i, 1] = 0.0
                out[j, i, 2] = 0.0
                continue

            surf_col = sphere_col[nearest_idx]
            ambient = 0.05 * surf_col
            cr, cg, cb = ambient[0], ambient[1], ambient[2]

            for Lidx in range(num_lights):
                Lpos = light_pos[Lidx]
                Lvec = Lpos - P
                dist = Lvec.norm()
                if dist < 1e-6:
                    continue
                L = Lvec / dist

                dot_nl = N.dot(L)
                if dot_nl <= 0.0:
                    continue

                H = normalize(L + V)

                # Shadow
                in_shadow = False
                if shadows_on:
                    eps = 1e-3
                    shadow_o = P + eps * N
                    shadow_dir = L
                    for s2 in range(num_spheres):
                        if s2 == nearest_idx:
                            continue
                        t_sh = intersect_sphere(shadow_o, shadow_dir, s2)
                        if t_sh > 0.0 and t_sh < dist - 1e-6:
                            in_shadow = True
                            break
                if in_shadow:
                    continue

                diff = kd * dot_nl
                spec = ks * (max(N.dot(H), 0.0) ** shininess)
                attenuation = light_I0[Lidx] / (dist * dist + 1e-6)
                Lcol = light_col[Lidx]

                diffuse = diff * attenuation * surf_col * Lcol
                specular = spec * attenuation * Lcol

                cr += diffuse[0] + specular[0]
                cg += diffuse[1] + specular[1]
                cb += diffuse[2] + specular[2]

            out[j, i, 0] = min(max(cr, 0.0), 1.0)
            out[j, i, 1] = min(max(cg, 0.0), 1.0)
            out[j, i, 2] = min(max(cb, 0.0), 1.0)


def render_scene_to_image(Wres, Hres):
    global img_field, IMG_W, IMG_H

    out_np = np.zeros((Hres, Wres, 3), dtype=np.float32)

    render_kernel(Wres, Hres, float(W_MM), float(H_MM), float(SCREEN_Z), out_np)

    maxv = out_np.max()
    if maxv <= 0:
        maxv = 1.0
    img8 = np.clip((out_np) * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img8, mode='RGB')
    return pil


class LR5App:
    def __init__(self, root):

        self._cam_update_after_id = None

        self.root = root
        root.title("ЛР5 — Taichi renderer (Blinn-Phong, shadows, color)")
        self.frame = ttk.Frame(root, padding=6)
        self.frame.pack(fill='both', expand=True)

        # Top: Notebook tabs
        self.nb = ttk.Notebook(self.frame)
        self.nb.pack(side='left', fill='y')

        # tabs
        self.tab_camera = ttk.Frame(self.nb, padding=6)
        self.tab_lights = ttk.Frame(self.nb, padding=6)
        self.tab_spheres = ttk.Frame(self.nb, padding=6)
        self.tab_params = ttk.Frame(self.nb, padding=6)
        self.tab_render = ttk.Frame(self.nb, padding=6)

        self.nb.add(self.tab_camera, text="Camera")
        self.nb.add(self.tab_lights, text="Lights")
        self.nb.add(self.tab_spheres, text="Spheres")
        self.nb.add(self.tab_params, text="Params")
        self.nb.add(self.tab_render, text="Render")

        # Right: render canvas
        self.view = ttk.Frame(self.frame, padding=6)
        self.view.pack(side='right', fill='both', expand=True)
        self.canvas_label = tk.Label(self.view, bg='gray15')
        self.canvas_label.pack(fill='both', expand=True)

        btns = ttk.Frame(self.view)
        btns.pack(pady=10)

        ttk.Button(btns, text="Render", command=self.render_and_update).pack(side='left', padx=5)
        ttk.Button(btns, text="Save PNG", command=self.save_image).pack(side='left', padx=5)

        self.spheres = []
        self.lights = []

        self.spheres.append({"pos": np.array([50.0, 50.0, 1000.0], dtype=np.float32), "R": 50.0,
                             "col": np.array([0.0, 1.0, 1.0], dtype=np.float32)})
        self.spheres.append({"pos": np.array([200.0, 200.0, 1000.0], dtype=np.float32), "R": 100.0,
                             "col": np.array([1.0, 0.0, 1.0], dtype=np.float32)})
        self.lights_strvar = tk.StringVar(value="[-300,-300,1000,1700, 150, 150, 0];[300,-300,1000,1000, 0, 100, 100];")
        self.kd = tk.DoubleVar(value=0.5)
        self.ks = tk.DoubleVar(value=0.8)
        self.shininess = tk.DoubleVar(value=200.0)
        self.Wres = tk.IntVar(value=800)
        self.Hres = tk.IntVar(value=800)
        self.shadows = tk.BooleanVar(value=True)

        # camera vars
        self.cam_x = tk.DoubleVar(value=448.0)
        self.cam_y = tk.DoubleVar(value=350.0)
        self.cam_z = tk.DoubleVar(value=3000.0)

        # build tabs
        self.build_camera_tab()
        self.build_lights_tab()
        self.build_spheres_tab()
        self.build_params_tab()
        self.build_render_tab()

        # initial sync
        self.sync_scene_to_taichi()
        # self.render_and_update()

    def schedule_camera_rerender(self):
        # отменяем предыдущий таймер, если есть
        if self._cam_update_after_id is not None:
            self.root.after_cancel(self._cam_update_after_id)

        # запускаем рендер через 80 мс (debounce)
        self._cam_update_after_id = self.root.after(80, self.render_and_update)

    def build_camera_tab(self):
        f = self.tab_camera
        ttk.Label(f, text="Camera position (always looks at 0,0,0)").pack(anchor='w')

        # === Кнопки ортогональных видов ===
        view_frame = ttk.LabelFrame(f, text="Ортогональные проекции")
        view_frame.pack(fill='x', pady=8)

        btn_front = ttk.Button(view_frame, text="Вид спереди", command=self.set_front_view)
        btn_top = ttk.Button(view_frame, text="Вид сверху", command=self.set_top_view)
        btn_side = ttk.Button(view_frame, text="Вид сбоку", command=self.set_side_view)

        btn_front.pack(side='left', expand=True, fill='x', padx=4, pady=4)
        btn_top.pack(side='left', expand=True, fill='x', padx=4, pady=4)
        btn_side.pack(side='left', expand=True, fill='x', padx=4, pady=4)

        # === Ручное управление камерой ===
        frm = ttk.Frame(f)
        frm.pack(anchor='w', pady=10)

        # ---- X ----
        ttk.Label(frm, text="X").grid(row=0, column=0, padx=5)
        tk.Scale(frm, variable=self.cam_x, from_=-5000, to=5000,
                 orient='horizontal', length=300,
                 command=lambda v: self.schedule_camera_rerender()
                 ).grid(row=0, column=1, padx=5)

        ex = ttk.Entry(frm, textvariable=self.cam_x, width=8)
        ex.grid(row=0, column=2, padx=5)
        ex.bind("<KeyRelease>", lambda e: self.schedule_camera_rerender())

        # ---- Y ----
        ttk.Label(frm, text="Y").grid(row=1, column=0, padx=5)
        tk.Scale(frm, variable=self.cam_y, from_=-5000, to=5000,
                 orient='horizontal', length=300,
                 command=lambda v: self.schedule_camera_rerender()
                 ).grid(row=1, column=1, padx=5)

        ey = ttk.Entry(frm, textvariable=self.cam_y, width=8)
        ey.grid(row=1, column=2, padx=5)
        ey.bind("<KeyRelease>", lambda e: self.schedule_camera_rerender())

        # ---- Z ----
        ttk.Label(frm, text="Z").grid(row=2, column=0, padx=5)
        tk.Scale(frm, variable=self.cam_z, from_=200, to=20000,
                 orient='horizontal', length=300,
                 command=lambda v: self.schedule_camera_rerender()
                 ).grid(row=2, column=1, padx=5)

        ez = ttk.Entry(frm, textvariable=self.cam_z, width=8)
        ez.grid(row=2, column=2, padx=5)
        ez.bind("<KeyRelease>", lambda e: self.schedule_camera_rerender())

    def reset_preview_camera(self):
        self.cam_az.set(0.0)
        self.cam_el.set(0.0)
        self.cam_dist.set(3000.0)
        self.render_and_update()

    def build_lights_tab(self):
        f = self.tab_lights
        ttk.Label(f, text="Lights (format: [x,y,z,I0] or [x,y,z,I0,R,G,B]; separate with ';')").pack(anchor='w')
        ent = ttk.Entry(f, textvariable=self.lights_strvar, width=70)
        ent.pack(anchor='w', pady=6)
        ttk.Button(f, text="Apply lights", command=self.apply_lights_from_text).pack(anchor='w')
        ttk.Label(f, text="Example: [-600,-500,1500,12000];[700,-400,1600,12000]").pack(anchor='w', pady=4)

    def build_spheres_tab(self):
        f = self.tab_spheres
        # fields to add sphere
        frm = ttk.Frame(f)
        frm.pack(anchor='w', pady=4)
        ttk.Label(frm, text="X").grid(row=0, column=0)
        self.sx = tk.DoubleVar(value=0.0)
        ttk.Entry(frm, textvariable=self.sx, width=8).grid(row=0, column=1)
        ttk.Label(frm, text="Y").grid(row=0, column=2)
        self.sy = tk.DoubleVar(value=0.0)
        ttk.Entry(frm, textvariable=self.sy, width=8).grid(row=0, column=3)
        ttk.Label(frm, text="Z").grid(row=0, column=4)
        self.sz = tk.DoubleVar(value=1000.0)
        ttk.Entry(frm, textvariable=self.sz, width=8).grid(row=0, column=5)
        ttk.Label(frm, text="R").grid(row=1, column=0)
        self.sR = tk.DoubleVar(value=100.0)
        ttk.Entry(frm, textvariable=self.sR, width=8).grid(row=1, column=1)
        ttk.Label(frm, text="Rcol").grid(row=1, column=2)
        self.scr = tk.DoubleVar(value=1.0)
        ttk.Entry(frm, textvariable=self.scr, width=6).grid(row=1, column=3)
        ttk.Label(frm, text="Gcol").grid(row=1, column=4)
        self.scg = tk.DoubleVar(value=1.0)
        ttk.Entry(frm, textvariable=self.scg, width=6).grid(row=1, column=5)
        ttk.Label(frm, text="Bcol").grid(row=1, column=6)
        self.scb = tk.DoubleVar(value=1.0)
        ttk.Entry(frm, textvariable=self.scb, width=6).grid(row=1, column=7)

        btns = ttk.Frame(f)
        btns.pack(anchor='w', pady=6)
        ttk.Button(btns, text="Add sphere", command=self.add_sphere_from_fields).pack(side='left', padx=4)
        ttk.Button(btns, text="Remove selected", command=self.remove_selected_sphere).pack(side='left')

        ttk.Label(f, text="Spheres in scene:").pack(anchor='w', pady=(8, 2))
        self.listbox = tk.Listbox(f, width=60, height=8)
        self.listbox.pack(anchor='w')
        self.refresh_spheres_listbox()

    def build_params_tab(self):
        f = self.tab_params
        ttk.Label(f, text="Lighting params").pack(anchor='w')
        ttk.Label(f, text="kd (diffuse)").pack(anchor='w')
        tk.Scale(f, variable=self.kd, from_=0.0, to=2.0, orient='horizontal', length=300, resolution=0.01).pack(
            anchor='w')
        ttk.Label(f, text="ks (specular)").pack(anchor='w')
        tk.Scale(f, variable=self.ks, from_=0.0, to=2.0, orient='horizontal', length=300, resolution=0.01).pack(
            anchor='w')
        ttk.Label(f, text="shininess").pack(anchor='w')
        tk.Scale(f, variable=self.shininess, from_=1, to=10000, orient='horizontal', length=300).pack(anchor='w')

        ttk.Separator(f, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(f, text="Shadows").pack(anchor='w')
        ttk.Checkbutton(f, text="Учитывать тени", variable=self.shadows).pack(anchor='w')

        ttk.Separator(f, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(f, text="Image resolution").pack(anchor='w')
        resf = ttk.Frame(f);
        resf.pack(anchor='w')
        ttk.Label(resf, text="W").grid(row=0, column=0)
        ttk.Entry(resf, textvariable=self.Wres, width=6).grid(row=0, column=1)
        ttk.Label(resf, text="H").grid(row=0, column=2)
        ttk.Entry(resf, textvariable=self.Hres, width=6).grid(row=0, column=3)

    def build_render_tab(self):
        f = self.tab_render
        ttk.Button(f, text="Render and Update", command=self.render_and_update, width=20).pack(pady=8)
        ttk.Button(f, text="Save last image (PNG)", command=self.save_image, width=20).pack(pady=6)
        ttk.Label(f, text="Notes: use 'Apply lights' after editing lights text.", foreground='gray').pack(anchor='w',
                                                                                                          pady=6)

    def add_sphere_from_fields(self):
        pos = np.array([float(self.sx.get()), float(self.sy.get()), float(self.sz.get())], dtype=np.float32)
        Rv = float(self.sR.get())
        col = np.array([float(self.scr.get()), float(self.scg.get()), float(self.scb.get())], dtype=np.float32)
        if len(self.spheres) >= MAX_SPHERES:
            messagebox.showwarning("Limit", f"Max spheres ({MAX_SPHERES}) reached.")
            return
        self.spheres.append({"pos": pos, "R": Rv, "col": col})
        self.refresh_spheres_listbox()

    def refresh_spheres_listbox(self):
        self.listbox.delete(0, tk.END)
        for s in self.spheres:
            p = s["pos"]
            Rv = s["R"]
            c = s["col"]
            self.listbox.insert(tk.END,
                                f"x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f}, R={Rv:.1f}, col=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")

    def remove_selected_sphere(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        del self.spheres[idx]
        self.refresh_spheres_listbox()

    def apply_lights_from_text(self):
        txt = self.lights_strvar.get()
        parsed = parse_lights(txt)
        if len(parsed) > MAX_LIGHTS:
            messagebox.showwarning("Limit", f"Max lights is {MAX_LIGHTS}. Truncating.")
            parsed = parsed[:MAX_LIGHTS]
        self.lights = parsed
        messagebox.showinfo("Lights", f"Applied {len(self.lights)} lights.")

    def sync_scene_to_taichi(self):
        n = len(self.spheres)
        n = clamp(n, 0, MAX_SPHERES)
        sphere_active[None] = n
        for i in range(MAX_SPHERES):
            if i < n:
                s = self.spheres[i]
                sphere_pos[i] = ti.Vector(list(s["pos"]))
                sphere_rad[i] = float(s["R"])
                col = s["col"]
                sphere_col[i] = ti.Vector(list(col))
            else:
                sphere_pos[i] = ti.Vector([0.0, 0.0, 0.0])
                sphere_rad[i] = 0.0
                sphere_col[i] = ti.Vector([0.0, 0.0, 0.0])

        # lights
        m = len(self.lights)
        m = clamp(m, 0, MAX_LIGHTS)
        light_active[None] = m
        for i in range(MAX_LIGHTS):
            if i < m:
                L = self.lights[i]
                light_pos[i] = ti.Vector(list(L["pos"]))
                light_I0[i] = float(L["I0"])
                light_col[i] = ti.Vector(list(L["col"]))
            else:
                light_pos[i] = ti.Vector([0.0, 0.0, 0.0])
                light_I0[i] = 0.0
                light_col[i] = ti.Vector([0.0, 0.0, 0.0])

        kd_field[None] = float(self.kd.get())
        ks_field[None] = float(self.ks.get())
        shininess_field[None] = float(self.shininess.get())
        shadows_field[None] = 1 if self.shadows.get() else 0

        cam_x_field[None] = float(self.cam_x.get())
        cam_y_field[None] = float(self.cam_y.get())
        cam_z_field[None] = float(self.cam_z.get())

    def render_and_update(self):
        try:
            parsed = parse_lights(self.lights_strvar.get())
            if parsed:
                self.lights = parsed[:MAX_LIGHTS]
        except Exception as e:
            messagebox.showwarning("Lights parse", f"Could not parse lights: {e}")

        self.sync_scene_to_taichi()

        W = int(self.Wres.get())
        H = int(self.Hres.get())
        if W <= 0 or H <= 0:
            messagebox.showerror("Resolution", "Wrong resolution.")
            return

        pil = render_scene_to_image(W, H)
        self.last_image = pil
        pil.save("АКГ_лр5_сферы_taichi.png")
        print("Saved АКГ_лр5_сферы_taichi.png")

        vw = self.view.winfo_width() or 800
        vh = self.view.winfo_height() or 600
        max_side = min(vw - 20, vh - 20, 900)
        scale = min(max_side / max(W, H), 1.0) if max(W, H) != 0 else 1.0
        if scale <= 0:
            scale = 1.0
        disp_w = int(W * scale)
        disp_h = int(H * scale)
        im_disp = pil.resize((max(1, disp_w), max(1, disp_h)), Image.NEAREST)
        photo = ImageTk.PhotoImage(im_disp)
        self.canvas_label.config(image=photo)
        self.canvas_label.image = photo

    def save_image(self):
        if hasattr(self, 'last_image') and self.last_image is not None:
            path = "АКГ_лр5_сферы_taichi.png"
            self.last_image.save(path)
            messagebox.showinfo("Saved", f"Saved {path}")
        else:
            messagebox.showinfo("No image", "Render first.")

    def set_front_view(self):
        """Вид спереди: XY плоскость, камера смотрит с +Z"""
        self.cam_x.set(0.0)
        self.cam_y.set(0.0)
        self.cam_z.set(2000.0)  # далеко по Z
        self.schedule_camera_rerender()

    def set_top_view(self):
        """Вид сверху: XZ плоскость, камера смотрит с +Y"""
        self.cam_x.set(0.0)
        self.cam_y.set(2000.0)  # сверху (отрицательный Y, если Y вверх)
        self.cam_z.set(0.0)  # чтобы видеть сцену
        self.schedule_camera_rerender()

    def set_side_view(self):
        """Вид сбоку: YZ плоскость, камера смотрит с +X"""
        self.cam_x.set(2000.0)
        self.cam_y.set(0.0)
        self.cam_z.set(0.0)
        self.schedule_camera_rerender()


def main():
    root = tk.Tk()
    app = LR5App(root)
    root.geometry("1200x720")
    root.mainloop()


if __name__ == "__main__":
    main()
