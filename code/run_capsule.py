""" Runs the destriping algorithm """

import os
from datetime import datetime
from glob import glob
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import dask
import numpy as np
import tifffile as tif
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)
from aind_smartspim_destripe import __version__, zarr_destriper
from aind_smartspim_destripe.utils import utils
from natsort import natsorted


def get_data_config(
    data_folder: str,
    processing_manifest_path: Optional[str] = "processing_manifest.json",
    data_description_path: Optional[str] = "data_description.json",
):
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: Optional[str]
        Path for the processing manifest

    data_description_path: Optional[str]
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    derivatives_dict = utils.read_json_as_dict(
        f"{data_folder}/{processing_manifest_path}"
    )
    data_description_dict = utils.read_json_as_dict(
        f"{data_folder}/{data_description_path}"
    )

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset


def generate_data_processing(
    channel_name: str,
    destripe_version: str,
    destripe_config: dict,
    start_time: datetime,
    end_time: datetime,
    output_directory: str,
):
    """
    Generates a destriping data processing
    for the processed channel.

    Paramters
    -----------
    channel_name: str
        SmartSPIM channel to process

    destripe_version: str
        Destriping version

    input_path: str
        Path where the images are located

    output_path: str
        Path where the images are stored

    destripe_config: dict
        Dictionary with the configuration
        for the destriping algorithm

    note_shadow_correction: str
        Shadow correction notes

    start_time: datetime
        Time the destriping process
        started

    end_time: datetime
        Time the destriping process
        ended

    output_directory: str
        Path where we want to store the
        processing manifest

    """
    output_directory = os.path.abspath(output_directory)

    if not os.path.exists(output_directory):
        raise FileNotFoundError(
            f"Please, check that this folder exists {output_directory}"
        )

    input_path = destripe_config["input_path"]
    output_path = destripe_config["output_path"]

    note_shadow_correction = "Applying the flats that come from the microscope"

    if destripe_config.get("retrospective"):
        note_shadow_correction = """The flats were computed from the data \
            with basicpy, these were applied with the destriping algorithm \
            and with the current dark from the microscope.
            """

    del destripe_config["input_path"]
    del destripe_config["output_path"]

    pipeline_process = PipelineProcess(
        data_processes=[
            DataProcess(
                name=ProcessName.IMAGE_DESTRIPING,
                software_version=destripe_version,
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(input_path),
                output_location=str(output_path),
                code_version=destripe_version,
                code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-destripe",
                parameters=destripe_config,
                notes=f"Destriping for channel {channel_name} in zarr format",
            ),
            DataProcess(
                name=ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
                software_version=destripe_version,
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(input_path),
                output_location=str(output_path),
                code_version=destripe_version,
                code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-destripe",
                parameters={},
                notes=note_shadow_correction,
            ),
        ],
        processor_full_name="Camilo Laiton",
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        pipeline_version="3.0.0",
    )

    processing = Processing(
        processing_pipeline=pipeline_process,
        notes="This processing only contains metadata about destriping \
        and needs to be compiled with other steps at the end",
    )

    with open(
        f"{output_directory}/image_destriping_{channel_name}_processing.json", "w"
    ) as f:
        f.write(processing.model_dump_json(indent=3))


def get_microscope_flats(
    channel_name: str, derivatives_folder: str
) -> Tuple[np.ndarray]:
    """
    Gets the microscope flats

    Parameters
    ----------
    channel_name : str
        Channel to be processed.

    derivatives_folder: str
        Path where the derivatives folder is.

    logger: logging.Logger
        Logging object

    Raises
    ------
    KeyError:
        Raises whenever we can't find the XY folders
        or brain side.

    Returns
    -------
    Tuple[List[ArrayLike], dictionary]
        Tuple with the flafields per brain hemisphere,
        current dark from the microscope and metadata.json
        content.
    """
    flatfield = None
    metadata_json = None

    waves = [p for p in channel_name.split("_") if p.isdigit()]

    metadata_json_path = derivatives_folder.joinpath("metadata.json")

    if metadata_json_path.exists() and len(waves):
        # If the flats exist, I can't apply the flats
        # without the metadata.json since I do not know which
        # brain hemisphere is correct for each flat

        orig_metadata_json = utils.read_json_as_dict(filepath=metadata_json_path)
        curr_emision_wave = int(waves[0])
        tile_config = orig_metadata_json.get("tile_config")
        metadata_json = {}

        if tile_config is None:
            raise ValueError("Please, verify metadata.json")

        # Getting only XY folders for the current emission wave
        # to know which locations used which flatfield
        for time_step, value in tile_config.items():
            config_em_wave = value.get("Laser")

            if int(config_em_wave) == curr_emision_wave:
                x_folder = value.get("X")
                y_folder = value.get("Y")
                brain_side = value.get("Side")  # 0 left hemisphere, 1 right hemisphere

                if x_folder is None or y_folder is None or brain_side is None:
                    raise KeyError("Please, check the data in metadata.json")

                if metadata_json.get(x_folder) is None:
                    metadata_json[x_folder] = {}

                metadata_json[x_folder][y_folder] = int(brain_side)

        # The flats are one per hemisphere, we need to check
        # metadata.json to know which tile is in which laser
        flatfield = [
            tif.imread(g)
            for g in natsorted(
                glob(f"{derivatives_folder}/FlatReal{curr_emision_wave}_*.tif")
            )
            if os.path.exists(g)
        ]

        # reading flatfields, we should have 2, one per brain hemisphere
        if len(flatfield) != 2:
            raise ValueError(
                f"Error while reading the microscope flatfields: {flatfield}"
            )

    return flatfield, metadata_json


def get_resolution(acquisition_config):
    # Grabbing a tile with metadata from acquisition - we assume all dataset
    # was acquired with the same resolution
    tile_coord_transforms = acquisition_config["tiles"][0]["coordinate_transformations"]

    scale_transform = [
        x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return x, y, z


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def run():
    """Validates parameters and runs the destriper"""

    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/acquisition.json",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    print(f"Data in folder: {list(data_folder.glob('*'))}")

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    dask.config.set({"distributed.worker.memory.terminate": False})

    BASE_PATH = data_folder
    acquisition_path = data_folder.joinpath("acquisition.json")

    acquisition_dict = utils.read_json_as_dict(acquisition_path)

    if not len(acquisition_dict):
        raise ValueError(
            f"Not able to read acquisition metadata from {acquisition_path}"
        )

    voxel_resolution = get_resolution(acquisition_dict)

    derivatives_path = data_folder.joinpath("derivatives")

    print(f"Derivatives path data: {list(derivatives_path.glob('*'))}")

    channels = [
        folder.name
        for folder in list(BASE_PATH.glob("Ex_*_Em_*"))
        if os.path.isdir(folder)
    ]
    laser_tiles_path = data_folder.joinpath("laser_tiles.json")

    if not laser_tiles_path.exists():
        raise FileNotFoundError(f"Path {laser_tiles_path} does not exist!")

    laser_tiles = utils.read_json_as_dict(str(laser_tiles_path))

    print(f"Laser tiles: {laser_tiles}")

    if len(channels):

        for channel_name in channels:
            estimated_channel_flats = natsorted(
                list(data_folder.glob(f"estimated_flat_laser_{channel_name}*.tif"))
            )

            if not len(estimated_channel_flats):
                raise FileNotFoundError(
                    f"Error while retrieving flats from the data folder for channel {channel_name}"
                )

            parameters = {
                "input_path": BASE_PATH.joinpath(channel_name),
                "output_path": str(results_folder),
                "no_cells_config": {
                    "wavelet": "db3",
                    "level": None,
                    "sigma": 128,
                    "max_threshold": 12,
                },
                "cells_config": {
                    "wavelet": "db3",
                    "level": None,
                    "sigma": 64,
                    "max_threshold": 3,
                },
                "retrospective": True,  # Default behavior
            }

            destriping_start_time = time()

            zarr_destriper.destripe_channel(
                zarr_dataset_path=BASE_PATH,
                channel_name=channel_name,
                results_folder=results_folder,
                derivatives_path=derivatives_path,
                xyz_resolution=voxel_resolution,
                estimated_channel_flats=estimated_channel_flats,
                laser_tiles=laser_tiles,
                parameters=parameters,
            )

            destriping_end_time = time()

            generate_data_processing(
                channel_name=channel_name,
                destripe_version=__version__,
                destripe_config=parameters,
                start_time=destriping_start_time,
                end_time=destriping_end_time,
                output_directory=results_folder,
            )

    else:
        print(f"No channels to process in {BASE_PATH}")


if __name__ == "__main__":
    run()
