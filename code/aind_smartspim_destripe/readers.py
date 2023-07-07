"""
Defining image reading interface
"""

from pathlib import Path
from typing import Union

import imageio as iio
import numpy as np
import tifffile

PathLike = Union[Path, str]

SUPPORTED_READING_EXTENSIONS = [".tif", ".tiff", ".raw", ".png"]


def _get_extension(path):
    """Extract the file extension from the provided path

    Parameters
    ----------
    path : str
        path with a file extension

    Returns
    -------
    ext : str
        file extension of provided path

    """
    return Path(path).suffix


def raw_imread(path):
    """Read a .raw file

    :param path: path to the file
    :returns: a Numpy read-only array mapping the file as an image
    """
    as_uint32 = np.memmap(path, dtype=">u4", mode="r", shape=(2,))
    width_be, height_be = as_uint32[:2]
    del as_uint32
    as_uint32 = np.memmap(path, dtype="<u4", mode="r", shape=(2,))
    width_le, height_le = as_uint32[:2]
    del as_uint32
    #
    # Heuristic, detect endianness by assuming that the smaller width is
    # the right one. Works for widths < 64K
    #
    if width_le < width_be:
        width, height = width_le, height_le
        dtype = "<u2"
    else:
        width, height = width_be, height_be
        dtype = ">u2"

    try:
        return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(width, height))
    except:
        print("Bad path: %s" % path)
        raise


def imread(path: PathLike) -> np.array:
    """Load a tiff or raw image

    Parameters
    ----------
    path : PathLike
        path to tiff or raw image

    Returns
    -------
    img : ndarray
        image as a numpy array

    """
    path = str(path)

    img = None
    extension = _get_extension(path)
    if extension == ".raw":
        img = raw_imread(path)
    elif extension == ".tif" or extension == ".tiff":
        img = tifffile.imread(path)
    elif extension == ".png":
        img = iio.imread(path)

    return img
