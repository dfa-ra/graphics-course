import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import re

# ==================== НАСТРОЙКИ ====================


# параметры по умолчанию
kd = 0.5
ks = 0.8
shininess = 200.0

Wres = 150
Hres = 100

center = np.array([0.0, 0.0, 1000.0])  # центр сферы
R = 300.0  # радиус сферы

W_mm = 500.00
H_mm = 300.0
screen_z = 0.0

initial_z = 3000.0
initial_lights = "[-600,-500,1500,12000];[700,-400,1600,12000]"

root = tk.Tk()
root.title("ЛР4 — Сфера Блинн-Фонг (камера + источники света)")

# ==================== ПЕРЕМЕННЫЕ GUI ====================
z_var = tk.DoubleVar(value=initial_z)
lights_str = tk.StringVar(value=initial_lights)

kd_var = tk.DoubleVar(value=kd)
ks_var = tk.DoubleVar(value=ks)
shininess_var = tk.DoubleVar(value=shininess)

Wres_var = tk.IntVar(value=Wres)
Hres_var = tk.IntVar(value=Hres)

# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ====================

def add_slider(label, var, frm, to, res):
    tk.Label(root, text=label, font=("Arial", 12)).pack(pady=(10, 2))
    fr = tk.Frame(root)
    fr.pack()
    tk.Scale(fr, from_=frm, to=to, resolution=res,
             orient='horizontal', length=500, variable=var).pack(side='left')
    tk.Entry(fr, textvariable=var, width=10).pack(side='left', padx=10)

# Z наблюдателя
add_slider("Z наблюдателя [мм]:", z_var, 800, 10000, 10)

# Разрешение
add_slider("Wres (горизонтальное разрешение):", Wres_var, 50, 500, 1)
add_slider("Hres (вертикальное разрешение):", Hres_var, 50, 500, 1)

# Коэффициенты модели освещения
add_slider("kd (диффузный коэффициент):", kd_var, 0.0, 2.0, 0.01)
add_slider("ks (зеркальный коэффициент):", ks_var, 0.0, 2.0, 0.01)
add_slider("shininess (блеск):", shininess_var, 1, 10000, 10)

# Источники света
tk.Label(root, text="Источники света (формат: [x,y,z,I0];...):",
         font=("Arial", 12)).pack(pady=(15, 5))

entry_lights = tk.Entry(root, textvariable=lights_str,
                        font=("Consolas", 11), width=70)
entry_lights.pack(pady=5)

btn = tk.Button(root, text="Пересчитать и обновить",
                font=("Arial", 14), height=2)
btn.pack(pady=10)

label_img = tk.Label(root, bg="gray20")
label_img.pack(padx=10, pady=10)


# ==================== ЛОГИКА ====================

def parse_lights(text):
    lights = []
    parts = text.replace(" ", "").split(';')
    for part in parts:
        if not part.strip():
            continue
        nums = re.findall(r'-?\d+\.?\d*', part)
        if len(nums) == 4:
            x, y, z, I0 = map(float, nums)
            lights.append({"pos": np.array([x, y, z]), "I0": I0})
    return lights


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def render(*args):
    global Wres, Hres, kd, ks, shininess

    # --- обновляем параметры ---
    Wres = int(Wres_var.get())
    Hres = int(Hres_var.get())

    kd = float(kd_var.get())
    ks = float(ks_var.get())
    shininess = float(shininess_var.get())

    z_obs = z_var.get()
    observer_pos = np.array([0.0, 0.0, z_obs])

    try:
        current_lights = parse_lights(lights_str.get())
        if not current_lights:
            print("Ошибка: не найдено ни одного источника света!")
            return
    except:
        print("Ошибка парсинга источников света!")
        return

    print(f"\nПересчёт при Z={z_obs:.1f}, Wres={Wres}, Hres={Hres}, "
          f"kd={kd}, ks={ks}, shininess={shininess}")

    brightness = np.zeros((Hres, Wres))
    max_b = 0.0
    min_b = np.inf

    for j in range(Hres):
        for i in range(Wres):
            pixel_size = max(W_mm / Wres, H_mm / Hres)
            screen_w = pixel_size * Wres
            screen_h = pixel_size * Hres

            x = -screen_w / 2 + (i + 0.5) * pixel_size
            y = screen_h / 2 - (j + 0.5) * pixel_size

            screen_pt = np.array([x, y, screen_z])
            dir_vec = screen_pt - observer_pos

            oc = observer_pos - center
            a = np.dot(dir_vec, dir_vec)
            b = 2 * np.dot(dir_vec, oc)
            c = np.dot(oc, oc) - R * R

            disc = b * b - 4 * a * c
            if disc < 0:
                continue

            sqrt_d = np.sqrt(disc)
            t = (-b - sqrt_d) / (2 * a)
            if t <= 0:
                t = (-b + sqrt_d) / (2 * a)
                if t <= 0:
                    continue

            P = observer_pos + t * dir_vec
            N = normalize(P - center)
            V = normalize(observer_pos - P)
            if np.dot(N, V) <= 0:
                continue

            bright = 0.0
            for light in current_lights:
                L_vec = light["pos"] - P
                dist = np.linalg.norm(L_vec)
                if dist < 1e-6:
                    continue

                L = L_vec / dist
                H = normalize(L + V)

                diff = kd * max(np.dot(N, L), 0)
                spec = ks * (max(np.dot(N, H), 0) ** shininess)

                bright += (diff + spec) * light["I0"] / (dist * dist)

            brightness[j, i] = bright
            max_b = max(max_b, bright)
            if 0 < bright < min_b:
                min_b = bright

    if max_b == 0:
        max_b = 1.0

    img_array = (brightness / max_b * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img_array, mode="L")
    im.save("АКГ_лр4_сфера.png")

    print("Изображение сохранено → АКГ_лр4_сфера.png")

    im_display = im.resize((5*Wres, 5*Hres), Image.NEAREST)
    photo = ImageTk.PhotoImage(im_display)
    label_img.config(image=photo)
    label_img.image = photo


btn.config(command=render)

render()
root.mainloop()
