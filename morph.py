from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_dilation, dilation

_SELEM = np.ones((3, 3), dtype=np.int64)

def apply_threshold(img, threshold=.5):
    """Applies the given threshold to an image, converting it into black and white"""
    result = np.ones_like(img, dtype=np.int64)
    result[np.abs(img) <= threshold] = 0
    result[np.abs(img) > threshold] = 1
    return result

def _selem_is_contained_in(window):
    """Returns True if ee is contained in win. Otherwise, returns False"""        

    idx_selem = np.where(_SELEM.ravel() == 1)[0]
    idx_win = np.where(window.ravel() == 1)[0]
    return set(idx_selem) <= set(idx_win)

def add_padding(img, radius):
    width, height = img.shape
    pad_img_shape = (width + radius - 1, height + radius - 1)
    pad_img = np.zeros(pad_img_shape).astype(np.float64)
    pad_img[radius-2:(width + radius-2), radius-2:(height + radius-2)] = img
    return pad_img

def process_pixel(i, j, operation, as_gray, img):
    radius = _SELEM.shape[0]
    neighbors = img[i:i+radius, j:j+radius]
    if as_gray:
        neighbors = np.delete(neighbors.flatten(), radius+1)
    return operation(neighbors, as_gray)


def _apply_filter(operation, img, as_gray, n_iterations, sel):
    """Applies a morphological operator a certain number of times (n_iterations) to an image"""
    global _SELEM
    _SELEM = sel
    img = img if as_gray else apply_threshold(img)
    width, height = img.shape
    prod = product(range(width), range(height))
    img_result = np.zeros_like(img)
    radius = _SELEM.shape[0]
    pad_img = add_padding(img, radius)
    if n_iterations >= 1:
        for i, j in prod:
            #if operation == 'er' and pad_img[i, j] == 1:
            #    img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
            #else:
            img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
        return _apply_filter(operation, img_result, as_gray, n_iterations-1, sel)
    return img

def _erosion(img, as_gray, n_iterations, sel):
    """Interface function to call erosion filter"""
    return _apply_filter(_apply_erosion, img, as_gray, n_iterations, sel)

def _dilation(img, as_gray, n_iterations, sel):
    """Interface function to call erosion filter"""
    return _apply_filter(_apply_dilation, img, as_gray, n_iterations, sel)

def _apply_erosion(neighbors, as_gray):
    """Modifies the current pixel value considering its neighbors
        and the erosion operation rules."""
    if not as_gray:
        if max(neighbors.ravel()) == 1:
            if _selem_is_contained_in(neighbors):
                return 1
        return 0
    return min(neighbors.ravel())

def _apply_dilation(neighbors, as_gray):
    """Modifies the current pixel value considering its neighbors
        and the dilation operation rules."""
    if not as_gray:
        if max(neighbors.ravel()) == 0:
            return 0
        return 1
    return max(neighbors.ravel())

def _opening(img, as_gray, n_iterations, sel):
    """Applies the opening operation"""
    eroded = _erosion(img, as_gray, n_iterations, sel)
    dilated = _dilation(eroded, as_gray, n_iterations, sel)
    return dilated

def _closing(img, as_gray, n_iterations, sel):
    """Applies the closing operation"""
    dilated = _dilation(img, as_gray, n_iterations, sel)
    return _erosion(dilated, as_gray, n_iterations, sel)

def _internal_gradient(img, as_gray, n_iterations, sel):
    """Applies the internal gradient operation"""
    img = img if as_gray else apply_threshold(img)
    return img - _erosion(img, as_gray, n_iterations, sel)

def _external_gradient(img, as_gray, n_iterations, sel):
    """Applies the external gradient operation"""
    img = img if as_gray else apply_threshold(img)
    return _dilation(img, as_gray, n_iterations, sel) - img

def _morphological_gradient(img, as_gray, n_iterations, sel):
    """Applies the morphological gradient operation"""
    dilated = _dilation(img, as_gray, n_iterations, sel)
    eroded = _erosion(img, as_gray, n_iterations, sel)
    return dilated - eroded

def _white_top_hat(img, as_gray, n_iterations, sel):
    """Applies the white top-hat operation"""
    if not as_gray:
        img = apply_threshold(img)
        wth = np.abs(_opening(img, as_gray, n_iterations, sel) - img)
        return apply_threshold(wth)
    return _opening(img, as_gray, n_iterations, sel) - img

def _black_top_hat(img, as_gray, n_iterations, sel):
    """Applies the black top-hat operation"""
    if not as_gray:
        img = apply_threshold(img)
        bth = np.abs(_closing(img, as_gray, n_iterations, sel) - img)
        return apply_threshold(bth)
    return _closing(img, as_gray, n_iterations, sel) - img

_OPS = {
    'er': _erosion,
    'di': _dilation,
    'op': _opening,
    'cl': _closing,
    'ig': _internal_gradient,
    'eg': _external_gradient,
    'mg': _morphological_gradient,
    'wth': _white_top_hat,
    'bth': _black_top_hat
}

def morph_filter(operator='er',
                 img=None,
                 sel=np.ones((3, 3), dtype=np.int64),
                 n_iterations=1,
                 as_gray=False):
    """Allows to apply multiple morphological operations over an image"""
    return _OPS[operator](img, as_gray, n_iterations, sel)
