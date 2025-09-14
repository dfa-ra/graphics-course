import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from config import IMAGE_PATHS


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Фото во весь экран с гистограммой RGB")

        # Размер экрана
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.geometry(f"{self.screen_width}x{self.screen_height}")

        # Label для картинки (на весь экран)
        self.image_label = tk.Label(root, bg="black")
        self.image_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Фрейм для кнопок (правый верхний угол)
        self.btn_frame = ttk.Frame(root)
        self.btn_frame.place(relx=1.0, y=10, anchor="ne")

        for i, path in enumerate(IMAGE_PATHS):
            btn = ttk.Button(self.btn_frame, text=f"Картинка {i+1}",
                             command=lambda p=path: self.show_image(p))
            btn.grid(row=0, column=i, padx=5)

        # Canvas для диаграммы (правый нижний угол)
        self.canvas = tk.Canvas(root, width=300, height=200, bg="white")
        self.canvas.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")

    def show_image(self, path):
        # Загружаем и растягиваем картинку на весь экран (без сохранения пропорций)
        img = Image.open(path)
        img = img.resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)

        # Отображаем фото
        self.image_label.configure(image=self.photo)

        # Считаем средние значения каналов
        r, g, b = self.get_average_rgb(img)

        # Рисуем диаграмму
        self.draw_chart(r, g, b)

    def get_average_rgb(self, img):
        pixels = list(img.getdata())
        r = sum(p[0] for p in pixels) / len(pixels)
        g = sum(p[1] for p in pixels) / len(pixels)
        b = sum(p[2] for p in pixels) / len(pixels)
        return r, g, b

    def draw_chart(self, r, g, b):
        self.canvas.delete("all")

        values = [r, g, b]
        colors = ["red", "green", "blue"]
        width = 60
        spacing = 20
        max_height = 150
        max_value = max(values)

        for i, (val, color) in enumerate(zip(values, colors)):
            height = (val / max_value) * max_height
            x0 = i * (width + spacing) + 30
            y0 = 180 - height
            x1 = x0 + width
            y1 = 180
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
            self.canvas.create_text((x0+x1)//2, y0-10, text=f"{int(val)}", fill=color)
