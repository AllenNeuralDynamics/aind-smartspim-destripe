""" Runs the destriping algorithm """
import json
import multiprocessing
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import aind_smartspim_destripe.flatfield_estimation as flat_est
import numpy as np
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)
from aind_smartspim_destripe import __version__, destriper
from aind_smartspim_destripe.utils import utils


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
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
):
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
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
    note_shadow_correction = "The flats were computed from the data with basicpy, these were applied with the destriping algorithm"

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


def run():
    """Validates parameters and runs the destriper"""

    no_cells_config = {
        "wavelet": "db3",
        "level": None,
        "sigma": 128,
        "max_threshold": 12,
    }
    cells_config = {"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3}

    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")

    # Dataset configuration in the processing_manifest.json
    pipeline_config, smartspim_dataset = get_data_config(data_folder=data_folder)

    print(f"Processing dataset {smartspim_dataset}")

    # Getting channel -> In the pipeline we must pass SmartSPIM/channel_name to data folder
    channel_name = glob(f"{data_folder}/*/")[0].split("/")[-2]
    input_path_str = f"{data_folder}/{channel_name}"
    input_path = Path(os.path.abspath(input_path_str))

    # Output path will be in /results/{channel_name}
    output_path = Path(results_folder).joinpath(f"{channel_name}")

    logger = utils.create_logger(output_log_path=results_folder)
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

    # Check if we have flat field and dark field

    # Estimating flat field and dark field
    folder_structure = utils.read_image_directory_structure(data_folder)

    shading_parameters = {
        "get_darkfield": True,
        "smoothness_flatfield": 1.0,
        "smoothness_darkfield": 20,
        "sort_intensity": True,
        "max_reweight_iterations": 35,
        # "resize_mode":"skimage_dask"
    }

    channel_path = list(folder_structure.keys())[0]
    cols = list(folder_structure[channel_path].keys())
    rows = [row for row in list(folder_structure[channel_path][cols[0]].keys())]
    n_cols = len(cols)
    n_rows = len(rows)
    len_stack = len(folder_structure[channel_path][cols[0]][rows[0]])

    slide_idxs = [len_stack // 5, len_stack // 3, len_stack // 2, int(len_stack // 1.5)]
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

    flats_dir = f"{results_folder}/flatfield_correction_{channel_name}"
    utils.create_folder(dest_dir=flats_dir)

    # Unifying fields with median
    for slide_idx, fields in shading_correction_per_slide.items():
        flatfields.append(fields["flatfield"])
        darkfields.append(fields["darkfield"])
        baselines.append(fields["baseline"])
        np.save(f"{flats_dir}/flatfield_{slide_idx}.npy", fields["flatfield"])
        np.save(f"{flats_dir}/darkfield_{slide_idx}.npy", fields["darkfield"])
        np.save(f"{flats_dir}/baseline_{slide_idx}.npy", fields["baseline"])

    mode = "median"
    logger.info(f"Unifying fields using {mode} mode.")
    flatfield, darkfield, baseline = flat_est.unify_fields(
        flatfields, darkfields, baselines, mode=mode
    )

    np.save(f"{flats_dir}/{mode}_flafield.npy", flatfield)
    np.save(f"{flats_dir}/{mode}_darkfield.npy", darkfield)
    np.save(f"{flats_dir}/{mode}_baseline.npy", baseline)

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
        "shadow_correction": {"flatfield": flatfield, "darkfield": darkfield},
    }

    destriping_start_time = datetime.now()

    if input_path.is_dir():
        logger.info("Starting destriping")
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
            results_folder,
            "smartspim_destripe",
        )


if __name__ == "__main__":
    run()
