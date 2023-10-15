""" Runs the destriping algorithm """
import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path

from aind_data_schema import Processing
from aind_data_schema.processing import (DataProcess, PipelineProcess,
                                         ProcessName)
from aind_smartspim_destripe import __version__, destriper


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

    del destripe_config["input_path"]
    del destripe_config["output_path"]

    destriping_process = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Camilo Laiton",
            pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
            pipeline_version="1.5.0",
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
            ],
        )
    )

    destriping_process.write_standard_file(
        output_directory=output_directory
    )


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

    # Getting channel -> In the pipeline we must pass SmartSPIM/channel_name to data folder
    channel_name = glob(f"{data_folder}/*/")[0].split("/")[-2]
    input_path_str = f"{data_folder}/{channel_name}"
    input_path = Path(os.path.abspath(input_path_str))

    # Output path will be in /results/{channel_name}
    output_path = Path(results_folder).joinpath(f"{channel_name}")

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
    }

    destriping_start_time = datetime.now()

    if input_path.is_dir():
        destriper.batch_filter(**parameters)

    destriping_end_time = datetime.now()

    generate_data_processing(
        channel_name=channel_name,
        destripe_version=__version__,
        destripe_config=parameters,
        start_time=destriping_start_time,
        end_time=destriping_end_time,
        output_directory=results_folder,
    )


if __name__ == "__main__":
    run()
