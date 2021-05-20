"""Utilities for reading and writing depth maps from/to disk."""

import re
import sys
from typing import Tuple

import numpy as np


def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb") # treat as binary and read-only
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf": # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename: str, image: np.ndarray, scale: float = 1) -> None:
    """Save a depth map from a .pfm file

    Args:
        filename: output .pfm file path string,
        image: depth map to save, of shape (H,W) or (H,W,C)
        scale: scale parameter to save
    """
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n".encode("utf-8") if color else "Pf\n".encode("utf-8"))
    file.write("{} {}\n".format(image.shape[1], image.shape[0]).encode("utf-8"))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write(("%f\n" % scale).encode("utf-8"))

    image.tofile(file)
    file.close()
