from PIL import ImageEnhance, ImageFilter
from .base import FilterRegistry


@FilterRegistry.register("Brightness")
def f_brightness(img, brightness=1.0, **_):
    img = img.convert("RGB")  # на всякий случай
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            r = min(255, max(0, int(r * brightness)))
            g = min(255, max(0, int(g * brightness)))
            b = min(255, max(0, int(b * brightness)))
            pixels[x, y] = (r, g, b)

    return img


@FilterRegistry.register("Contrast")
def f_contrast(img, contrast=1.0, **_):
    img = img.convert("RGB")
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            r = min(255, max(0, int((r - 128) * contrast + 128)))
            g = min(255, max(0, int((g - 128) * contrast + 128)))
            b = min(255, max(0, int((b - 128) * contrast + 128)))
            pixels[x, y] = (r, g, b)

    return img


@FilterRegistry.register("Blur")
def f_blur(img, blur_radius=2, **_):
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
