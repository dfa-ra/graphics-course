import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from filters import FilterRegistry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

IMAGE_PATH = "source/img.jpg"


class ImageProcessor:
    def __init__(self, img):
        self.original = img.convert("RGB")
        self.processed = self.original.copy()

    def apply(self, filters, params):
        img = self.original.copy()
        for name in filters:
            func = FilterRegistry.get_filters().get(name)
            if func:
                img = func(img, **params)
        self.processed = img
        return img


class ImageEditorApp:
    def __init__(self, image_path):
        self.processor = ImageProcessor(Image.open(image_path))
        self.root = tk.Tk()
        self.root.title("🖼️ Image Filters — Modular Edition")
        self.root.minsize(1100, 750)
        self._build_ui()
        self._bind_events()
        self.update_preview()

    # === Построение интерфейса ===
    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # === Панель управления ===
        controls = ttk.Frame(main, padding=10)
        controls.grid(row=0, column=0, rowspan=2, sticky="ns")
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=3)
        main.grid_rowconfigure(1, weight=2)

        ttk.Label(controls, text="🎨 Фильтры:", font=("Arial", 11, "bold")).grid(sticky="w", pady=(0, 5))
        self.listbox = tk.Listbox(controls, selectmode="multiple", height=10, exportselection=False)
        for name in FilterRegistry.names():
            self.listbox.insert("end", name)
        self.listbox.grid(sticky="we", pady=(0, 8))

        # Слайдеры параметров
        def add_slider(text, var, from_, to_, val):
            ttk.Label(controls, text=text).grid(sticky="w", pady=(4, 0))
            scale = ttk.Scale(controls, from_=from_, to=to_, orient="horizontal")
            scale.set(val)
            scale.grid(sticky="we", pady=(0, 6))
            setattr(self, var, scale)

        add_slider("Blur:", "blur_scale", 0, 10, 2)
        add_slider("Contrast:", "contrast_scale", 0.1, 3, 1)
        add_slider("Brightness:", "brightness_scale", 0.1, 3, 1)
        add_slider("Posterize bits:", "posterize_scale", 1, 8, 4)

        # Кнопки
        btns = ttk.Frame(controls)
        btns.grid(sticky="we", pady=(6, 6))
        for i in range(3):
            btns.grid_columnconfigure(i, weight=1)

        ttk.Button(btns, text="Применить", command=self.apply_filters).grid(row=0, column=0, sticky="we")
        ttk.Button(btns, text="Сброс", command=self.reset).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(btns, text="Сохранить", command=self.save).grid(row=0, column=2, sticky="we")

        ttk.Button(controls, text="Выбрать всё", command=lambda: self.listbox.select_set(0, "end")).grid(
            sticky="we", pady=(6, 2))
        ttk.Button(controls, text="Снять всё", command=lambda: self.listbox.select_clear(0, "end")).grid(
            sticky="we")

        # === Область изображений ===
        img_frame = ttk.Frame(main)
        img_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        img_frame.grid_columnconfigure((0, 1), weight=1)
        img_frame.grid_rowconfigure(0, weight=1)

        self.left = tk.Canvas(img_frame, bg="#ddd")
        self.right = tk.Canvas(img_frame, bg="#ddd")
        self.left.grid(row=0, column=0, sticky="nsew", padx=6)
        self.right.grid(row=0, column=1, sticky="nsew", padx=6)

        ttk.Label(img_frame, text="Исходное").grid(row=1, column=0)
        ttk.Label(img_frame, text="Результат").grid(row=1, column=1)

        # === Гистограмма внизу ===
        hist_frame = ttk.Frame(main)
        hist_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 10))
        hist_frame.grid_rowconfigure(0, weight=1)
        hist_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.figure, master=hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)

        self._left_img = self._right_img = None

    # === События ===
    def _bind_events(self):
        self.root.bind("<Configure>", lambda e: self.root.after(300, self.update_preview))
        self.listbox.bind("<Double-Button-1>", lambda e: self.apply_filters())

    # === Применить фильтры ===
    def apply_filters(self):
        params = dict(
            blur_radius=self.blur_scale.get(),
            contrast=self.contrast_scale.get(),
            brightness=self.brightness_scale.get(),
            posterize_bits=int(round(self.posterize_scale.get())),
        )
        selected = [self.listbox.get(i) for i in self.listbox.curselection()]
        self.processor.apply(selected, params)
        self.update_preview()

    # === Сброс ===
    def reset(self):
        self.processor.processed = self.processor.original.copy()
        self.listbox.selection_clear(0, "end")
        self.blur_scale.set(2)
        self.contrast_scale.set(1)
        self.brightness_scale.set(1)
        self.posterize_scale.set(4)
        self.update_preview()

    # === Сохранение ===
    def save(self):
        try:
            self.processor.processed.save("result.png")
            messagebox.showinfo("Сохранено", "✅ result.png сохранён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # === Обновление изображений ===
    def update_preview(self):
        def fit(img, w, h):
            iw, ih = img.size
            s = min(w / iw, h / ih)
            return img.resize((int(iw * s), int(ih * s)))

        lw, lh = self.left.winfo_width(), self.left.winfo_height()
        rw, rh = self.right.winfo_width(), self.right.winfo_height()
        if lw < 5 or lh < 5 or rw < 5 or rh < 5:
            self.root.after(200, self.update_preview)
            return

        left = fit(self.processor.original, lw, lh)
        right = fit(self.processor.processed, rw, rh)
        self._left_img = ImageTk.PhotoImage(left)
        self._right_img = ImageTk.PhotoImage(right)

        self.left.delete("all")
        self.right.delete("all")
        self.left.create_image(lw // 2, lh // 2, image=self._left_img, anchor="center")
        self.right.create_image(rw // 2, rh // 2, image=self._right_img, anchor="center")

        self.update_histogram(self.processor.processed)

    # === Гистограмма RGB (столбчатая) ===
    def update_histogram(self, img):
        """Обновляет большую RGB-гистограмму в виде бар-графика"""
        self.ax.clear()
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            colors = ("red", "green", "blue")
            labels = ("Red", "Green", "Blue")
            bins = np.arange(257)

            # Рисуем три столбчатых гистограммы с прозрачностью
            for i, (col, label) in enumerate(zip(colors, labels)):
                hist, _ = np.histogram(arr[..., i], bins=bins)
                self.ax.bar(
                    bins[:-1],
                    hist,
                    color=col,
                    alpha=0.5,
                    width=1.0,
                    label=label,
                )

            self.ax.set_xlim(0, 256)
            self.ax.set_xlabel("Яркость (0–255)", fontsize=9)
            self.ax.set_ylabel("Количество пикселей", fontsize=9)
            self.ax.legend(loc="upper right", fontsize=8)
            self.ax.set_title("Столбчатая гистограмма каналов RGB", fontsize=11, fontweight="bold")
            self.ax.grid(alpha=0.3)
            self.figure.tight_layout()
            self.canvas_hist.draw()

    # === Запуск ===
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ImageEditorApp(IMAGE_PATH)
    app.run()
