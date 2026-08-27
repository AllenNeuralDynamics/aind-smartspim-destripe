"""Runs the destriping algorithm"""

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import dask
import numpy as np
import tifffile as tif
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import (DataProcess, ProcessName,
                                                ProcessStage)
from natsort import natsorted
from log_schema import setup_logging

from aind_smartspim_destripe import (__maintainers__, __pipeline_name__,
                                      __pipeline_version__, __title__,
                                      __url__, __version__, zarr_destriper)
from aind_smartspim_destripe.utils import metadata_compat, utils

logger = logging.getLogger(__name__)


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

    process_name = f"{__title__}"

    setup_logging(
        model={
            "pipeline_name": __pipeline_name__,
            "process_name": process_name,
            "software_name": __title__,
            "software_version": __version__,
        }
    )

    start_time = time.monotonic()
    dataset_name = None

    try:
        data_folder = Path(os.path.abspath("../data"))
        results_folder = Path(os.path.abspath("../results"))
        scratch_folder = Path(os.path.abspath("../scratch"))

        logger.info(
            "Destriping started",
            extra={
                "event_type": "stage_start",
                "data_folder": str(data_folder),
                "results_folder": str(results_folder),
            },
        )

        # It is assumed that these files
        # will be in the data folder
        required_input_elements = [
            f"{data_folder}/acquisition.json",
            f"{data_folder}/data_description.json",
        ]

        missing_files = validate_capsule_inputs(required_input_elements)

        logger.debug(f"Data in folder: {list(data_folder.glob('*'))}")

        if len(missing_files):
            raise ValueError(
                f"We miss the following files in the capsule input: {missing_files}"
            )

        dask.config.set({"distributed.worker.memory.terminate": False})

        # Make this a parameter
        bucket_name = "aind-open-data"

        acquisition_path = data_folder.joinpath("acquisition.json")
        acquisition_dict = utils.read_json_as_dict(acquisition_path)

        data_description_path = data_folder.joinpath("data_description.json")
        data_description_dict = utils.read_json_as_dict(data_description_path)

        if not len(acquisition_dict):
            raise ValueError(
                f"Not able to read acquisition metadata from {acquisition_path}"
            )

        if not len(data_description_dict):
            raise ValueError(
                f"Not able to read data description metadata from {data_description_path}"
            )

        voxel_resolution = metadata_compat.get_voxel_resolution(acquisition_dict)

        derivatives_path = data_folder.joinpath("derivatives")

        logger.debug(f"Derivatives path data: {list(derivatives_path.glob('*'))}")

        channels = None
        dataset_name = data_description_dict.get("name")

        # Dispatcher generates preprocess_{channel_name}.json files
        # These are split to instantiate a single machine per channel
        # Find channel configuration files using multiple patterns
        channel_config_paths = list(data_folder.glob("preprocess_*.json"))

        if not channel_config_paths:
            raise FileNotFoundError(
                "No preprocess_*.json configuration file found in data folder"
            )

        # The connection is default, so we can pick the first config
        BASE_PATH = data_folder
        if Path(channel_config_paths[0]).suffix == ".json":
            BASE_PATH = f"s3://{bucket_name}/"

        if utils.is_s3_path(str(BASE_PATH)):
            prefix = f"{dataset_name}/SPIM"
            BASE_PATH = f"{BASE_PATH}{prefix}"

            channel_config = utils.read_json_as_dict(channel_config_paths[0])
            channel_to_process = channel_config.get('channel')

            if not channel_to_process:
                raise ValueError(f"Please, provide a channel to process. Config: {channel_config_paths[0]}")

            channels = [
                i
                for i in utils.list_s3_folders(bucket=bucket_name, prefix=prefix)
                if str(channel_to_process) in i
            ]
        else:
            BASE_PATH = Path(BASE_PATH)
            channels = [
                folder.name
                for folder in list(BASE_PATH.glob("Ex_*_Em_*"))
                if os.path.isdir(folder)
            ]

        laser_tiles_path = data_folder.joinpath("laser_tiles.json")

        if not laser_tiles_path.exists():
            raise FileNotFoundError(f"Path {laser_tiles_path} does not exist!")

        laser_tiles = utils.read_json_as_dict(str(laser_tiles_path))

        logger.debug(f"Laser tiles: {laser_tiles}")

        logger.info(
            f"Destriping configuration resolved for dataset {dataset_name}",
            extra={
                "dataset_name": dataset_name,
                "derivatives_path": str(derivatives_path),
                "base_path": str(BASE_PATH),
                "channels": channels,
                "voxel_resolution": voxel_resolution,
            },
        )

        data_processes = []
        cpu_cores = utils.get_cpu_limit()

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
                    "input_path": f"{BASE_PATH}/{channel_name}",
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

                note_shadow_correction = (
                    "Applying the flats that come from the microscope"
                )

                if parameters.get("retrospective"):
                    note_shadow_correction = """The flats were computed from the data \
                    with basicpy, these were applied with the destriping algorithm \
                    and with the current dark from the microscope.
                    """

                channel_start_time = datetime.now(timezone.utc)
                resource_monitor = utils.ResourceMonitor(interval_seconds=30.0).start()

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

                resource_monitor.stop()
                channel_end_time = datetime.now(timezone.utc)

                channel_resources = resource_monitor.to_resource_usage(
                    cpu_cores=cpu_cores
                )
                channel_code = Code(
                    url=__url__, name=__title__, version=__version__
                )
                channel_duration_seconds = (
                    channel_end_time - channel_start_time
                ).total_seconds()

                data_processes.append(
                    DataProcess(
                        process_type=ProcessName.IMAGE_DESTRIPING,
                        name=f"Image destriping - {channel_name}",
                        stage=ProcessStage.PROCESSING,
                        code=channel_code,
                        experimenters=__maintainers__,
                        pipeline_name=__pipeline_name__,
                        start_date_time=channel_start_time,
                        end_date_time=channel_end_time,
                        output_path=str(results_folder),
                        output_parameters={
                            k: v
                            for k, v in parameters.items()
                            if k not in ("input_path", "output_path")
                        }
                        | {
                            "input_location": str(parameters["input_path"]),
                            "duration_seconds": channel_duration_seconds,
                        },
                        resources=channel_resources,
                        notes=f"Destriping for channel {channel_name} in zarr format",
                    )
                )

                data_processes.append(
                    DataProcess(
                        process_type=ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
                        name=f"Flatfield correction - {channel_name}",
                        stage=ProcessStage.PROCESSING,
                        code=channel_code,
                        experimenters=__maintainers__,
                        pipeline_name=__pipeline_name__,
                        start_date_time=channel_start_time,
                        end_date_time=channel_end_time,
                        output_path=str(results_folder),
                        output_parameters={
                            "input_location": str(parameters["input_path"]),
                            "duration_seconds": channel_duration_seconds,
                        },
                        resources=channel_resources,
                        notes=note_shadow_correction,
                    )
                )

        else:
            logger.warning(f"No channels to process in {BASE_PATH}")

        utils.generate_processing(
            data_processes=data_processes,
            dest_processing=results_folder,
            pipeline_name=__pipeline_name__,
            pipeline_version=__pipeline_version__,
            pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        )

        processing_json = results_folder / "processing.json"
        for channel_name in channels:
            shutil.copy(
                str(processing_json),
                str(results_folder / f"image_destriping_{channel_name}_processing.json"),
            )

        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.info(
            "Destriping completed",
            extra={
                "event_type": "stage_complete",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )

    except Exception:
        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.error(
            "Destriping failed",
            exc_info=True,
            extra={
                "event_type": "stage_failure",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
        raise


if __name__ == "__main__":
    run()
