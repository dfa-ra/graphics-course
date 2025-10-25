from PIL import ImageFilter

from .base import FilterRegistry


@FilterRegistry.register("Emboss")
def f_emboss(img, **_):
    return img.filter(ImageFilter.EMBOSS)
