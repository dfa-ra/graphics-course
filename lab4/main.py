import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import re

# ==================== НАСТРОЙКИ ====================
center = np.array([0.0, 0.0, 1000.0])  # центр сферы
R = 20.0  # радиус сферы

kd = 0.5
ks = 0.8
shininess = 200.0

Wres = Hres = 100
W_mm = H_mm = 2000.0
screen_z = 0.0

initial_z = 3000.0
initial_lights = "[-600,-500,1500,12000];[700,-400,1600,12000]"  # можно менять здесь

root = tk.Tk()
root.title("ЛР4 — Сфера Блинн-Фонг (камера + источники света)")

z_var = tk.DoubleVar(value=initial_z)
lights_str = tk.StringVar(value=initial_lights)

tk.Label(root, text="Z наблюдателя [мм]:", font=("Arial", 12)).pack(pady=(15, 5))
frame_z = tk.Frame(root)
frame_z.pack(pady=5)

tk.Scale(frame_z, from_=800, to=10000, resolution=10, orient='horizontal',
         length=500, variable=z_var).pack(side='left')
tk.Entry(frame_z, textvariable=z_var, width=10).pack(side='left', padx=10)

tk.Label(root, text="Источники света (формат: [x,y,z];[x,y,z];...):", font=("Arial", 12)).pack(pady=(20, 5))
entry_lights = tk.Entry(root, textvariable=lights_str, font=("Consolas", 11), width=70)
entry_lights.pack(pady=5)

btn = tk.Button(root, text="Пересчитать и обновить", font=("Arial", 14), height=2)
label_img = tk.Label(root, bg="gray20")
label_img.pack(padx=10, pady=10)


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


def render(*args):
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

    print(f"\nПересчёт при Z = {z_obs:.1f} мм, источников: {len(current_lights)}")

    brightness = np.zeros((Hres, Wres))
    max_b = 0.0
    min_b = np.inf

    for j in range(Hres):
        for i in range(Wres):
            x = -W_mm / 2 + (i + 0.5) * W_mm / Wres
            y = H_mm / 2 - (j + 0.5) * H_mm / Hres
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
            if bright > max_b:
                max_b = bright
            if bright > 0 and bright < min_b:
                min_b = bright

    if max_b == 0:
        max_b = 1.0
    img_array = (brightness / max_b * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img_array, mode="L")
    im.save("АКГ_лр4_сфера.png")

    display_size = 800
    im_display = im.resize((display_size, display_size), Image.NEAREST)

    print(f"Изображение сохранено → АКГ_лр4_сфера.png")

    min_b = min_b if min_b != np.inf else 0.0
    print(f"Макс. яркость (абс.): {max_b:.6f}")
    print(f"Мин. яркость (абс.):  {min_b:.6f}")

    def calc_at(p):
        N = normalize(p - center)
        V = normalize(observer_pos - p)
        if np.dot(N, V) <= 0:
            return 0.0
        b = 0.0
        for light in current_lights:
            L_vec = light["pos"] - p
            d = np.linalg.norm(L_vec)
            if d < 1e-6:
                continue
            L = L_vec / d
            H = normalize(L + V)
            diff = kd * max(np.dot(N, L), 0)
            spec = ks * (max(np.dot(N, H), 0) ** shininess)
            b += (diff + spec) * light["I0"] / (d * d)
        return b

    p1 = center + np.array([0, 0, R])
    p2 = center + np.array([R, 0, 0])
    p3 = center + np.array([0, R, 0])

    print(f"Яркость (0,0,+R): {calc_at(p1):.6f}")
    print(f"Яркость (+R,0,0): {calc_at(p2):.6f}")
    print(f"Яркость (0,+R,0): {calc_at(p3):.6f}")

    photo = ImageTk.PhotoImage(im_display)
    label_img.config(image=photo)
    label_img.image = photo


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def on_change(*args):
    render()


btn.config(command=render)
btn.pack(pady=10)

entry_lights.bind('<Return>', on_change)
root.bind('<Return>', lambda e: render())

render()

root.mainloop()