from PIL import ImageEnhance, ImageFilter
from .base import FilterRegistry


@FilterRegistry.register("Brightness")
def f_brightness(img, brightness=1.0, **_):
    return ImageEnhance.Brightness(img).enhance(brightness)


@FilterRegistry.register("Contrast")
def f_contrast(img, contrast=1.0, **_):
    return ImageEnhance.Contrast(img).enhance(contrast)


@FilterRegistry.register("Blur")
def f_blur(img, blur_radius=2, **_):
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
