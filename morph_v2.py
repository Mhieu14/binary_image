from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_dilation, dilation

_SELEM = np.ones((3, 3), dtype=np.int64)
_WHITE = 255 # white value

def add_padding(img, radius):
    width, height = img.shape
    pad_img_shape = (width + radius - 1, height + radius - 1)
    pad_img = np.zeros(pad_img_shape).astype(np.float64)
    k = int(radius/2)
    pad_img[k:width+k, k:height+k] = img
    return pad_img

def process_pixel_pad(i, j, operation, img, sel):
    radius = sel.shape[0]
    neighbors = img[i:i+radius, j:j+radius]
    return operation(neighbors, sel)

def _apply_filter(operation, img, dt_value, sel):
    width, height = img.shape
    prod = product(range(width), range(height))
    img_result = np.zeros_like(img)
    radius = sel.shape[0]
    logic_image = np.zeros_like(img)
    logic_image[img == dt_value] = 1
    pad_img = add_padding(logic_image, radius)
    for i, j in prod:
        logic_image[i, j] = process_pixel_pad(i, j, operation, pad_img, sel)
    return np.uint8(logic_image * dt_value)

def _apply_erosion(neighbors, sel):
    idx_selem = np.where(sel.ravel() == 1)[0]
    idx_win = np.where(neighbors.ravel() == 1)[0]
    return set(idx_selem) <= set(idx_win)

def _apply_dilation(neighbors, sel):
    idx_selem = np.where(sel.ravel() == 1)[0]
    idx_win = np.where(neighbors.ravel() == 1)[0]
    return not set(idx_selem).isdisjoint(idx_win)
    # return max(neighbors.ravel())
    # if max(neighbors.ravel()) == 0:
    #     return 0
    # return 1

def _erosion(img, dt_value, sel):
    return _apply_filter(_apply_erosion, img, dt_value, sel)

def _dilation(img, dt_value, sel):
    return _apply_filter(_apply_dilation, img, dt_value, sel)

def _opening(img, dt_value, sel):
    eroded = _erosion(img, dt_value, sel)
    dilated = _dilation(eroded, dt_value, sel)
    return dilated

def _closing(img, dt_value, sel):
    dilated = _dilation(img, dt_value, sel)
    return _erosion(dilated, dt_value, sel)

def _hit_and_miss(img, dt_value, sel):
    B1 = np.zeros_like(sel)
    B2 = np.zeros_like(sel)
    B1[sel == 1] = 1
    B2[sel == -1] = 1
    imgC = np.ones_like(img) * dt_value - img
    hit = _erosion(img, dt_value, B1)
    miss = _erosion(imgC, dt_value, B2)
    hit_and_miss = np.zeros_like(img)
    hit_and_miss[hit + miss > dt_value] = dt_value
    return hit_and_miss
    
_OPS = {
    'er': _erosion,
    'di': _dilation,
    'op': _opening,
    'cl': _closing,
    'h&m': _hit_and_miss
}

def morph_filter(operator='er',
                 img=None,
                 sel=np.ones((3, 3), dtype=np.int64),
                 dt_value=255):
    """Allows to apply multiple morphological operations over an image"""
    return _OPS[operator](img, dt_value, sel)
