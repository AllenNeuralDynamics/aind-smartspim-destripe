""" Runs the destriping algorithm """
import json
import os
from glob import glob
from pathlib import Path

from aind_smartspim_destripe import destriper


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


def run():
    """Validates parameters and runs the destriper"""

    no_cells_config = {
        "wavelet": "db3",
        "level": None,
        "sigma": 256,
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

    # Output path will be in /data/{channel_name}
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

    if input_path.is_dir():
        destriper.batch_filter(**parameters)


if __name__ == "__main__":
    run()
