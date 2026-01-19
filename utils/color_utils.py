from skimage import color
import numpy as np


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_array):
    rgb = np.asarray(rgb_array).reshape(-1)
    assert rgb.shape[0] == 3
    r, g, b = rgb
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def rgb_to_normalized_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)

    squeeze = False
    if rgb.ndim == 1:
        rgb = rgb[None, :]  # (N, 3)
        squeeze = True

    rgb = rgb.copy() / 255.0

    lab = color.rgb2lab(rgb)  # (N, 3)
    lab[:, 0] /= 100.0  # L ∈ [0, 1]
    lab[:, 1:] /= 128.0  # a,b ∈ [-1, 1]

    if squeeze:
        lab = lab[0]

    return lab


def single_hex_list_to_lab_arr(hex_list):
    rgb_list = [hex_to_rgb(h) for h in hex_list]
    lab_array = rgb_to_normalized_lab(np.array(rgb_list))
    return lab_array


def normalized_lab_to_rgb(lab):
    lab = np.asarray(lab, dtype=np.float32)

    squeeze = False
    if lab.ndim == 1:
        lab = lab[None, :]  # (N, 3)
        squeeze = True

    lab = lab.copy()
    lab[:, 0] *= 100.0  # L
    lab[:, 1:] *= 128.0  # a, b

    rgb = color.lab2rgb(lab)
    rgb = (rgb * 255.0).round().astype(np.uint8)

    if squeeze:
        rgb = rgb[0]  # back to (3,)

    return rgb
