import PIL
from PIL import Image
import numpy as np

def concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def blend(img, img_ori, mask):
    img = np.array(img) * (np.array(mask) / 255) + np.array(img_ori) * (1 - np.array(mask) / 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img
