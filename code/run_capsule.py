""" Runs the destriping algorithm """
import os
from pathlib import Path

from aind_smartspim_destripe import destriper


def run():
    """Validates parameters and runs the destriper"""

    no_cells_config = {"wavelet": "db3", "level": None, "sigma": 128, "max_threshold": 12}

    cells_config = {"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3}

    results_folder = os.path.abspath("../results")

    input_path_str = "/data/SmartSPIM_656020_2023-06-02_20-05-39/SmartSPIM/Ex_488_Em_525/483090/483090_713010"
    input_path = Path(os.path.abspath(input_path_str))

    output_path = Path(results_folder).joinpath(
        input_path_str.replace("/data/", "") + "_destriped"
    )

    parameters = {
        "input_path": input_path,
        "output_path": output_path,
        "workers": 16,
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
