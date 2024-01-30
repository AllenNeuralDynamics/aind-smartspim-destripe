""" Runs the destriping algorithm """
import json
import logging
import multiprocessing
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import aind_smartspim_destripe.flatfield_estimation as flat_est
import numpy as np
import tifffile as tif
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)
from aind_smartspim_destripe import __version__, destriper
from aind_smartspim_destripe.filtering import invert_image, normalize_image
from aind_smartspim_destripe.utils import utils
from natsort import natsorted


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.
    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.
    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.
    """

    dictionary = {}

    if os.path.exists(filepath):
        try:
            with open(filepath) as json_file:
                dictionary = json.load(json_file)

        except UnicodeDecodeError:
            print("Error reading json with utf-8, trying different approach")
            # This might lose data, verify with Jeff the json encoding
            with open(filepath, "rb") as json_file:
                data = json_file.read()
                data_str = data.decode("utf-8", errors="ignore")
                dictionary = json.loads(data_str)

    #             print(f"Reading {filepath} forced: {dictionary}")

    return dictionary


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

    derivatives_dict = read_json_as_dict(f"{data_folder}/{processing_manifest_path}")
    data_description_dict = read_json_as_dict(f"{data_folder}/{data_description_path}")

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
    shadow_correction_params = destripe_config["shadow_correction"]

    note_shadow_correction = "Using the flats that come from the microscope"

    if shadow_correction_params.get("retrospective"):
        note_shadow_correction = """The flats were computed from the data \
            with basicpy, these were applied with the destriping algorithm \
            and with the current dark from the microscope.
            """

    del destripe_config["input_path"]
    del destripe_config["output_path"]
    del destripe_config["shadow_correction"]

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
                notes=f"Destriping for channel {channel_name} in {destripe_config['output_format']} format",
            ),
            DataProcess(
                name=ProcessName.IMAGE_FLATFIELD_CORRECTION,
                software_version=destripe_version,
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(input_path),
                output_location=str(output_path),
                code_version=destripe_version,
                code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-destripe",
                parameters=shadow_correction_params,
                notes=note_shadow_correction,
            ),
        ],
        processor_full_name="Camilo Laiton",
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        pipeline_version="1.5.0",
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


def get_retrospective_flatfield_correction(
    data_folder: str,
    flats_dir: str,
    no_cells_config: dict,
    cells_config: dict,
    shading_parameters: dict,
    logger: logging.Logger,
) -> Tuple[np.ndarray]:
    """
    Estimates the flatfields from 4 slides distributed
    along the entire volume.

    Parameters
    ---------
    data_folder: str
        Folder where the channel data is located.

    flats_dir: str
        Path where the estimated flats will be written.

    no_cells_config: dict
        Dictionary with the configuration to filter the
        tiles that have no cells in them.

    cells_config: dict
        Dictionary with the configuration to filter the
        tiles that have cells in them.

    shading_parameters: dict
        Parameters to estimate the flatfields

    logger: logging.Logger
        Logging object

    Returns
    -------
    Tuple[np.ndarray]
        Tuple with the estimated flatfield,
        darkfield and baselines to correct
        the images
    """
    # Estimating flat field and dark field
    folder_structure = utils.read_image_directory_structure(data_folder)

    channel_path = list(folder_structure.keys())[0]
    cols = list(folder_structure[channel_path].keys())
    rows = [row for row in list(folder_structure[channel_path][cols[0]].keys())]
    n_cols = len(cols)
    n_rows = len(rows)
    len_stack = len(folder_structure[channel_path][cols[0]][rows[0]])

    slide_idxs = [
        len_stack // 5,
        len_stack // 3,
        len_stack // 2,
        int(len_stack // 1.5),
    ]
    logger.info(f"Using slides {slide_idxs} for flatfield and darkfield estimation")

    # Estimating flatfields and darkfields per slide
    shading_correction_per_slide = flat_est.slide_flat_estimation(
        folder_structure,
        channel_path,
        slide_idxs,
        shading_parameters,
        no_cells_config,
        cells_config,
    )

    flatfields = []
    darkfields = []
    baselines = []

    # Unifying fields with median
    for slide_idx, fields in shading_correction_per_slide.items():
        flatfields.append(fields["flatfield"])
        darkfields.append(fields["darkfield"])
        baselines.append(fields["baseline"])

        tif.imwrite(f"{flats_dir}/flatfield_{slide_idx}.tif", fields["flatfield"])
        tif.imwrite(f"{flats_dir}/darkfield_{slide_idx}.tif", fields["darkfield"])
        tif.imwrite(f"{flats_dir}/baseline_{slide_idx}.tif", fields["baseline"])

    mode = "median"
    logger.info(f"Unifying fields using {mode} mode.")
    flatfield, darkfield, baseline = flat_est.unify_fields(
        flatfields, darkfields, baselines, mode=mode
    )

    tif.imwrite(f"{flats_dir}/{mode}_flafield.tif", flatfield)
    tif.imwrite(f"{flats_dir}/{mode}_darkfield.tif", darkfield)
    tif.imwrite(f"{flats_dir}/{mode}_baseline.tif", baseline)

    return flatfield, darkfield, baseline


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

        orig_metadata_json = read_json_as_dict(filepath=metadata_json_path)
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
        print("Converting to 16bit")

        # reading flatfields, we should have 2, one per brain hemisphere
        if len(flatfield) != 2:
            raise ValueError(
                f"Error while reading the microscope flatfields: {flatfield}"
            )

    return flatfield, metadata_json


def run():
    """Validates parameters and runs the destriper"""

    no_cells_config = {
        "wavelet": "db3",
        "level": None,
        "sigma": 128,
        "max_threshold": 12,
    }
    cells_config = {"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3}

    # Check if we have flat field and dark field
    darkfield = None
    flatfield = None
    tile_config = None  # Used when the flats come from the microscope
    retrospective = False
    apply_microscope_flats = False  # If we want to apply the flats from the microscope
    shading_parameters = {}

    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")

    # Dataset configuration in the processing_manifest.json
    pipeline_config, smartspim_dataset = get_data_config(data_folder=data_folder)

    print(f"Processing dataset {smartspim_dataset}")

    # Getting channel -> In the pipeline we must pass SmartSPIM/channel_name to data folder
    channel_name = glob(f"{data_folder}/*/")[0].split("/")[-2]
    derivatives_folder = Path(f"{data_folder}/derivatives")
    input_path_str = f"{data_folder}/{channel_name}"
    input_path = Path(os.path.abspath(input_path_str))

    metadata_flats_dir = f"{results_folder}/flatfield_correction_{channel_name}"
    utils.create_folder(dest_dir=metadata_flats_dir)

    # Output path will be in /results/{channel_name}
    output_path = Path(results_folder).joinpath(f"{channel_name}")

    logger = utils.create_logger(output_log_path=metadata_flats_dir)
    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    if os.path.exists(derivatives_folder):
        logger.info("Using flat-fields from the microscope")

        # Reading darkfield
        darkfield_path = derivatives_folder.joinpath("DarkMaster.tif")

        try:
            darkfield = tif.imread(str(darkfield_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Please, provide the current dark from the microscope! Provided path: {darkfield_path}"
            )

        if apply_microscope_flats:
            flatfield, tile_config = get_microscope_flats(
                channel_name=channel_name,
                derivatives_folder=derivatives_folder,
            )
            # Normalizing and inverting flatfields from the microscope
            flatfield = normalize_image(flatfield)
            flatfield = invert_image(flatfield)

        else:
            logger.info("Ignoring microscope flats")

    if flatfield is None or tile_config is None:
        logger.info("Estimating flats with BasicPy...")
        shading_parameters = {
            "get_darkfield": True,
            "smoothness_flatfield": 1.0,
            "smoothness_darkfield": 20,
            "sort_intensity": True,
            "max_reweight_iterations": 35,
            # "resize_mode":"skimage_dask"
        }

        flatfield, basicpy_darkfield, baseline = get_retrospective_flatfield_correction(
            data_folder=data_folder,
            flats_dir=metadata_flats_dir,
            no_cells_config=no_cells_config,
            cells_config=cells_config,
            shading_parameters=shading_parameters,
            logger=logger,
        )
        retrospective = True

    shading_parameters["retrospective"] = retrospective

    parameters = {
        "input_path": input_path,
        "output_path": output_path,
        "workers": 32,
        "chunks": 1,
        "high_int_filt_params": cells_config,
        "low_int_filt_params": no_cells_config,
        "compression": 1,
        "output_format": ".tiff",
        "output_dtype": None,
        "shadow_correction": {
            "retrospective": retrospective,
            "flatfield": flatfield,  # Estimated with basicpy or using the flats from the microscope
            "darkfield": darkfield,  # Coming from the microscope
            "tile_config": tile_config,
        },
    }

    destriping_start_time = datetime.now()
    if input_path.is_dir():
        logger.info(
            f"Starting destriping and flatfielding with restrospective approach? {retrospective}"
        )
        destriper.batch_filter(**parameters)

    destriping_end_time = datetime.now()

    # Overwriting shadow correction estimated fields with shading parameters
    # To save them in processing.json
    parameters["shadow_correction"] = shading_parameters
    generate_data_processing(
        channel_name=channel_name,
        destripe_version=__version__,
        destripe_config=parameters,
        start_time=destriping_start_time,
        end_time=destriping_end_time,
        output_directory=results_folder,
    )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_flats_dir,
            "smartspim_destripe",
        )


if __name__ == "__main__":
    run()
