from PIL import Image, ImageOps
from .base import FilterRegistry


@FilterRegistry.register("Grayscale")
def f_grayscale(img, **_):
    return ImageOps.grayscale(img).convert("RGB")


@FilterRegistry.register("Invert")
def f_invert(img, **_):
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        inv = ImageOps.invert(Image.merge("RGB", (r, g, b)))
        r2, g2, b2 = inv.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    return ImageOps.invert(img)


@FilterRegistry.register("Sepia")
def f_sepia(img, **_):
    img = img.convert("RGB")
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pixels[x, y]
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
    return img


@FilterRegistry.register("Posterize")
def f_posterize(img, posterize_bits=4, **_):
    bits = max(1, min(int(posterize_bits), 8))
    return ImageOps.posterize(img.convert("RGB"), bits)
