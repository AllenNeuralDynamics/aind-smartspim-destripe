"""
SmartSPIM image filtering to remove
streaks
"""

import logging
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pywt
import tqdm
from scipy import fftpack
from skimage import filters

from .filtering import filter_steaks
from .readers import *

PathLike = Union[Path, str]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SUPPORTED_OUTPUT_EXTENSIONS = [".tif", ".tiff", ".png"]


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


def imsave(path, img, compression=1, output_format: Optional[str] = None):
    """Save an array as a TIFF, RAW or PNG

    The file format will be inferred from the file extension in `path` or
    by using the output_format extension

    Parameters
    ----------
    path : str
        path to tiff or raw image
    img : ndarray
        image as a numpy array
    compression : int
        compression level for tiff writing
    output_format : Optional[str]
        Desired format extension to save the image. Default: None
        Accepted ['.tiff', '.tif', '.png']
    """
    extension = _get_extension(path)

    if output_format is None:
        # Saving any input format to tiff
        if extension == ".raw" or extension == ".png":
            # TODO: get raw writing to work
            # raw.raw_imsave(path, img)
            # tifffile.imsave(os.path.splitext(path)[0]+'.tiff', img, compress=compression) # Use with versions <= 2020.9.3
            tifffile.imsave(
                os.path.splitext(path)[0] + ".tiff",
                img,
                compressionargs={"level": compression},
            )  # Use with version 2023.03.21

        elif extension == ".tif" or extension == ".tiff":
            # tifffile.imsave(path, img, compress=compression) # Use with versions <= 2020.9.3
            tifffile.imsave(
                path, img, compressionargs={"level": compression}
            )  # Use with version 2023.03.21

    else:
        # Saving output images based on the output format
        if output_format not in SUPPORTED_OUTPUT_EXTENSIONS:
            raise ValueError(
                f"Output format {output_format} is not valid! Supported extensions are: {SUPPORTED_OUTPUT_EXTENSIONS}"
            )

        filename = os.path.splitext(path)[0] + output_format
        if output_format == ".tif" or output_format == ".tiff":
            # tifffile.imsave(filename, img, compress=compression) # Use with versions <= 2020.9.3
            tifffile.imsave(
                path, img, compressionargs={"level": compression}
            )  # Use with version 2023.03.21

        elif output_format == ".png":
            # print(img.dtype)
            # iio.imwrite(filename, img, compress_level=compression) # Works fine up to version 2.15.0
            iio.v3.imwrite(filename, img, compress_level=compression)  # version 2.27.0


def read_filter_save(
    output_dir: PathLike,
    input_path: PathLike,
    output_path: PathLike,
    high_int_filter_params: dict,
    low_int_filter_params: dict,
    compression: int = 1,
    output_format: str = None,
    output_dtype=None,
):
    # Number of retries per file
    n = 3
    for i in range(n):
        try:
            # File must be in the SUPORTED_READING_EXTENSIONS
            # defined in the readers script
            raw_image = imread(input_path)
            dtype = raw_image.dtype
            if output_dtype is not None and isinstance(output_dtype, type):
                dtype = output_dtype

        except:
            if i == n - 1:
                file_name = os.path.join(output_root_dir, "destripe_log.txt")
                if not os.path.exists(file_name):
                    error_file = open(file_name, "w")
                    error_file.write(
                        "Error reading the following images.  We will interpolate their content."
                    )
                    error_file.close()
                error_file = open(file_name, "a+")
                error_file.write("\n{}".format(str(input_path)))
                error_file.close()
                return
            else:
                time.sleep(0.05)
                continue

    filtered_image = filter_steaks(
        image=raw_image,
        no_cells_config=low_int_filter_params,
        cells_config=high_int_filter_params,
    )

    nb_retry = 10
    # Save image, retry if OSError for NAS
    for _ in range(nb_retry):
        try:
            imsave(
                output_path,
                filtered_image.astype(dtype),
                compression=compression,
                output_format=output_format,
            )
        except OSError:
            logger.error(f"Retrying writing image in {output_path}...")
            continue
        break


def _read_filter_save(input_dict: dict):
    """Same as `read_filter_save' but with a single input dictionary. Used for pool.imap() in batch_filter

    Parameters
    ----------
    input_dict : dict
        input dictionary with arguments for `read_filter_save`.

    """
    read_filter_save(**input_dict)


def _find_all_images(
    search_path: PathLike, input_path: PathLike, output_path: PathLike
):
    """Find all images with a supported file extension within a directory and all its subdirectories

    Parameters
    ----------
    input_path : PathLike
        root directory to start image search

    Returns
    -------
    img_paths : List[PathLike]
        A list of Path objects for all found images

    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    search_path = Path(search_path)

    assert search_path.is_dir()
    img_paths = []
    for p in search_path.iterdir():
        if p.is_file():
            if p.suffix in SUPPORTED_READING_EXTENSIONS:
                img_paths.append(p)

        elif p.is_dir():
            rel_path = p.relative_to(input_path)
            o = output_path.joinpath(rel_path)
            if not o.exists():
                o.mkdir(parents=True)
            img_paths.extend(_find_all_images(p, input_path, output_path))

    return img_paths


def batch_filter(
    input_path: PathLike,
    output_path: PathLike,
    workers: int,
    chunks: int,
    high_int_filt_params: dict,
    low_int_filt_params: dict,
    compression: int = 1,
    output_format: str = None,
    output_dtype: type = None,
):
    error_path = os.path.join(output_path, "destripe_log.txt")
    if os.path.exists(error_path):
        os.remove(error_path)

    logger.info(f"Looking for images in {input_path}")
    img_paths = _find_all_images(input_path, input_path, output_path)
    logger.info(f"Found {len(img_paths)} compatible images")

    # copy text and ini files
    for file in input_path.iterdir():
        if Path(file).suffix in [".txt", ".ini"]:
            output_file = os.path.join(output_path, os.path.split(file)[1])
            shutil.copyfile(file, output_file)

    logger.info(f"Setting up {workers} workers...")

    args = []
    for p in img_paths:
        rel_path = p.relative_to(input_path)
        o = output_path.joinpath(rel_path)

        if not o.parent.exists():
            o.parent.mkdir(parents=True)

        arg_dict = {
            "output_dir": output_path,
            "input_path": p,
            "output_path": o,
            "high_int_filter_params": high_int_filt_params,
            "low_int_filter_params": low_int_filt_params,
            "compression": compression,
            "output_format": output_format,
            "output_dtype": output_dtype,
        }
        args.append(arg_dict)

    logger.info("Starting batch filtering")

    with multiprocessing.Pool(workers) as pool:
        list(
            tqdm.tqdm(
                pool.imap(_read_filter_save, args, chunksize=chunks),
                total=len(args),
                ascii=True,
            )
        )

    logger.info("Done with batch filtering!")

    if os.path.exists(error_path):
        logger.error(f"An error happened, see destripe log for more details")
