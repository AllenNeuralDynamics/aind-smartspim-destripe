"""
Estimates the flatfield, darkfield and baseline
that will be used to correct image tiles.
"""

from typing import List, Optional

import numpy as np
from basicpy import BaSiC
from skimage.io import imread

from .filtering import filter_stripes


def shading_correction(
    slides: List[np.array], shading_parameters: dict, mask: Optional[np.array] = None
):
    """
    Computes shading correction for each of the
    provided tiles for further post-processing.

    Parameters
    ----------
    slides: List[List[ArrayLike]]
        List of tiles per slide used to compute
        the shading fitting.

    shading_parameters: dict
        Parameters to build the basicpy object

    mask: ArrayLike
        Mask with weights for each of the pixels
        that determines the contribution of the fields
        to remove the shadows.

    Returns
    -------
    Tuple[Dict]
        tuple with the flatfield, darkfield and
        baseline results from the shadow fitting
        for further post-processing.
    """
    shading_obj = BaSiC(**shading_parameters)
    shading_results = []
    shading_obj.fit(images=np.array(slides), fitting_weight=mask)
    shading_results = {
        "flatfield": shading_obj.flatfield,
        "darkfield": shading_obj.darkfield,
        "baseline": shading_obj.baseline,
    }

    return shading_results


def unify_fields(
    flatfields: List[np.array],
    darkfields: List[np.array],
    baselines: List[np.array],
    mode: Optional[str] = "median",
):
    """
    Unifies the computed flatfields, darkfields and
    baselines using an statistical mode.

    Parameters
    ----------
    flatfields: List[np.array]
        List of computed flatfields per slide.

    darkfields: List[np.array]
        List of computed darkfields per slide.

    baselines: List[np.array]
        List of computed baselines per slide.

    mode: Optional[str]
        Statistical mode to combine flatfields,
        darkfields and baselines.

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        Combined flatfield, darkfield and baseline.
    """
    flatfield = None
    darkfield = None
    baseline = None

    flatfields = np.array(flatfields)
    darkfields = np.array(darkfields)
    baselines = np.array(baselines)

    if mode == "median":
        flatfield = np.median(flatfields, axis=0)
        darkfield = np.median(darkfields, axis=0)
        baseline = np.median(baselines, axis=0)

    elif mode == "mean":
        flatfield = np.mean(flatfields, axis=0)
        darkfield = np.mean(darkfields, axis=0)
        baseline = np.mean(baselines, axis=0)

    elif mode == "mip":
        flatfield = np.max(flatfields, axis=0)
        darkfield = np.min(darkfields, axis=0)
        baseline = np.max(baselines, axis=0)

    else:
        msg = "Accepted values are: ['mean', 'median', 'mip']"
        raise NotImplementedError(msg)

    flatfield = flatfield.astype(
        np.float16
    )  # np.clip(flatfield, 0, 65535).astype('uint16')
    darkfield = darkfield.astype(
        np.float16
    )  # np.clip(darkfield, 0, 65535).astype('uint16')
    baseline = baseline.astype(
        np.float16
    )  # np.clip(baseline, 0, 65535).astype('uint16')

    return flatfield, darkfield, baseline


def slide_flat_estimation(
    dict_struct: dict,
    channel_name: str,
    slide_idxs: List[int],
    shading_parameters: dict,
    no_cells_config,
    cells_config,
) -> dict:
    """
    Estimates the flatfields, darkfields and
    baselines using a list of slides that
    then will be combined into a single
    flatfield, darkfield and baseline.

    Parameters
    ----------
    dict_struct: dict
        Dictionary with the folder structure
        of the SmartSPIM dataset

    channel_name: str
        Channel name in the folder structure
        where the tiles are located.

    slide_idxs: List[int]
        List of slides that will be used
        to estimate the flatfields, darkfields
        and baselines.

    shading_parameters: dict
        Shading parameters used to instantiate
        the basicpy object.

    Returns
    -------
    dict
        Dictionary with the flatfields, darkfields
        and baselines for the slides.
    """
    dict_struct = dict_struct[channel_name]
    # channel_paths = list(dict_struct.keys())
    cols = list(dict_struct.keys())
    rows = [row.split("_")[-1] for row in list(dict_struct[cols[0]].keys())]
    row_name = f"{cols[0]}_{rows[0]}"

    # imgs = []
    shading_correction_per_slide = {}
    names = []
    # all_slides = []
    for slide_idx in slide_idxs:
        slide_name = dict_struct[cols[0]][row_name][slide_idx]
        slide_tiles = []
        for col in cols:
            for row in rows:
                row_col = f"{col}/{col}_{row}/{slide_name}"
                names.append(f"{col}_{row}")
                input_tile_path = f"{channel_name}/{row_col}"
                data = imread(input_tile_path)
                data_destriped = filter_stripes(
                    image=data,
                    input_tile_path=input_tile_path,
                    no_cells_config=no_cells_config,
                    cells_config=cells_config,
                )
                slide_tiles.append(data_destriped)

        shading_correction_per_slide[slide_idx] = shading_correction(
            slides=slide_tiles, shading_parameters=shading_parameters
        )
        shading_correction_per_slide[slide_idx]["data"] = slide_tiles

    return shading_correction_per_slide
