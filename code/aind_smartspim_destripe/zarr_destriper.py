import json
import logging
import multiprocessing
import os
from glob import glob
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Type, cast
import dask

import dask.array as da
import numpy as np
import psutil
import tifffile as tif
import xarray_multiscale
import zarr
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)
from aind_large_scale_prediction._shared.types import ArrayLike, PathLike
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from dask.distributed import Client, LocalCluster, performance_report
from natsort import natsorted
from numcodecs import blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata
from scipy.ndimage import binary_fill_holes, grey_dilation, map_coordinates
from skimage.measure import regionprops

import filtering as fl
from blocked_zarr_writer import BlockedArrayWriter
from utils import utils


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


def generate_data_processing(
    channel_name: str,
    destripe_version: str,
    destripe_config: dict,
    start_time,
    end_time,
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


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def extract_global_to_local(
    global_ids_with_cells: ArrayLike,
    global_slices: Tuple[slice],
    pad: Optional[int] = 0,
) -> ArrayLike:
    """
    Takes global ZYX positions and converts them
    into local ZYX position within the chunk shape.
    It is important to provide the chunk of data with
    overlapping area in each direction to pick cell
    centroids out of the current boundary.

    Parameters
    ----------
    global_ids_with_cells: ArrayLike
        Global ZYX cell centroids with cell ids
        in the last dimension.

    global_slices: Tuple[slice]
        Global coordinate position of this chunk of
        data in the global image.

    pad: Optional[int]
        Padding applied when computing the flows,
        centroids and histograms. Default: 0

    Returns
    -------
    ArrayLike:
        ZYX positions of centroids within the current
        chunk of data.
    """

    start_pos = []
    stop_pos = []

    for c in global_slices:
        start_pos.append(c.start - pad)
        stop_pos.append(c.stop + pad)

    start_pos = np.array(start_pos)
    stop_pos = np.array(stop_pos)

    # Picking locations within current chunk area in global space
    picked_global_ids_with_cells = global_ids_with_cells[
        (global_ids_with_cells[:, 0] >= start_pos[0])
        & (global_ids_with_cells[:, 0] < stop_pos[0])
        & (global_ids_with_cells[:, 1] >= start_pos[1])
        & (global_ids_with_cells[:, 1] < stop_pos[1])
        & (global_ids_with_cells[:, 2] >= start_pos[2])
        & (global_ids_with_cells[:, 2] < stop_pos[2])
    ]

    # Mapping to the local coordinate system of the chunk
    picked_global_ids_with_cells[..., :3] = (
        picked_global_ids_with_cells[..., :3] - start_pos - pad
    )

    # Validating seeds are within block boundaries
    picked_global_ids_with_cells = picked_global_ids_with_cells[
        (picked_global_ids_with_cells[:, 0] >= 0)
        & (picked_global_ids_with_cells[:, 0] <= (stop_pos[0] - start_pos[0]) + pad)
        & (picked_global_ids_with_cells[:, 1] >= 0)
        & (picked_global_ids_with_cells[:, 1] <= (stop_pos[1] - start_pos[1]) + pad)
        & (picked_global_ids_with_cells[:, 2] >= 0)
        & (picked_global_ids_with_cells[:, 2] <= (stop_pos[2] - start_pos[2]) + pad)
    ]

    return picked_global_ids_with_cells


def execute_worker(
    data,
    batch_super_chunk,
    batch_internal_slice,
    cells_config,
    no_cells_config,
    overlap_prediction_chunksize,
    output_destriped_zarr,
    shadow_correction,
    dataset_name,
    logger: logging.Logger,
):

    data = np.squeeze(data, axis=0)

    (
        global_coord_pos,
        global_coord_positions_start,
        global_coord_positions_end,
    ) = recover_global_position(
        super_chunk_slice=batch_super_chunk,  # sample.batch_super_chunk[0],
        internal_slices=batch_internal_slice,  # sample.batch_internal_slice,
    )

    unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
        global_coord_pos=global_coord_pos,
        block_shape=data.shape,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        dataset_shape=output_destriped_zarr.shape,
    )
    # Converting to 5D
    unpadded_local_slice = list(
        (
            slice(0, 1),
            slice(0, 1),
        )
        + unpadded_local_slice
    )

    # Checking borders
    output_slices = list(
        (
            slice(0, 1),
            slice(0, 1),
        )
        + unpadded_global_slice
    )

    for idx in range(output_destriped_zarr.ndim):
        if output_slices[idx].stop > output_destriped_zarr.shape[idx]:
            rest = output_slices[idx].stop - output_destriped_zarr.shape[idx]
            unpadded_local_slice[idx] = slice(
                unpadded_local_slice[idx].start, unpadded_local_slice[idx].stop - rest
            )
            output_slices[idx] = slice(
                output_slices[idx].start, output_destriped_zarr.shape[idx]
            )

    output_slices = tuple(output_slices)
    unpadded_local_slice = tuple(unpadded_local_slice)

    filtered_data = np.zeros_like(data)

    input_tile_path = dataset_name.replace(".zarr", "")
    #     print("Input tile path: ", dataset_name, input_tile_path)

    for plane_idx in range(data.shape[-3]):
        filtered_data[plane_idx, ...] = fl.filter_stripes(
            image=data[plane_idx, ...],
            input_tile_path=input_tile_path,
            no_cells_config=no_cells_config,
            cells_config=cells_config,
            shadow_correction=shadow_correction,
            microscope_high_int=2500,
        )

    filtered_data = pad_array_n_d(
        arr=filtered_data[unpadded_local_slice[2:]], dim=output_destriped_zarr.ndim
    )

    filtered_data_converted = np.clip(filtered_data, 0, 65535).astype(np.uint16)
    #     filtered_data_converted = (filtered_data_converted / filtered_data.max() * 65535).astype(np.uint16)
    
    output_destriped_zarr[output_slices] = filtered_data
    
#     return filtered_data_converted, output_slices


def _execute_worker(params):
    """
    Worker interface to provide parameters
    """
    return execute_worker(**params)


def helper_schedule_jobs(picked_blocks, pool, logger):

    # Assigning blocks to execution workers
    jobs = [
        pool.apply_async(_execute_worker, args=(picked_block,))
        for picked_block in picked_blocks
    ]

    logger.info(f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs")

    # Wait for all processes to finish
    results = [job.get() for job in jobs]  # noqa: F841

    return results


def compute_pyramid(
    data,
    n_lvls,
    scale_axis,
    chunks="auto",
):
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mean,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid]


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translation is not None:
        transforms[0].append({"type": "translation", "translation": translation})
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [f"Channel:{image_name}:{i}" for i in range(data_shape[1])]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr: da.Array,
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr : array-like
        The input array.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    print("WRITING METADATA")
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr.shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, arr.chunksize, arr.shape, None
    )
    fmt.validate_coordinate_transformations(
        arr.ndim, n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def compute_multiscale(
    output_zarr,
    zarr_group,
    scale_factor,
    n_workers,
    voxel_size,
    image_name,
    n_levels=3,
    threads_per_worker=1,
):

    # Instantiating local cluster for parallel writing
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)
    #     performance_report_path = f"/results/report.html"

    start_time = time()
    pyramid_group = output_zarr

    previous_scale = da.from_zarr(pyramid_group, output_zarr.chunks)

    written_pyramid = []

    if np.issubdtype(previous_scale.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(np.uint16).min,
            np_info_func(np.uint16).max,
        )
        for _ in range(previous_scale.shape[1])
    ]

    # Setting values for SmartSPIM
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 350.0) for _ in range(previous_scale.shape[1])]

    # Writing OME-NGFF metadata
    write_ome_ngff_metadata(
        group=zarr_group,
        arr=previous_scale,
        image_name=image_name,
        n_lvls=n_levels,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_names=[image_name],
        channel_colors=[0x690AFE],
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=None,
    )

    # Writing zarr and performance report
    #     with performance_report(filename=performance_report_path):
    for i in range(1, n_levels):

        if i != 1:
            previous_scale = da.from_zarr(pyramid_group, output_zarr.chunks)

        # Writing zarr
        block_shape = list(
            BlockedArrayWriter.get_block_shape(
                arr=previous_scale, target_size_mb=12800  # 51200,
            )
        )

        # Formatting to 5D block shape
        block_shape = ([1] * (5 - len(block_shape))) + block_shape

        new_scale_factor = (
            [1] * (len(previous_scale.shape) - len(scale_factor))
        ) + scale_factor

        previous_scale_pyramid = compute_pyramid(
            data=previous_scale,
            scale_axis=new_scale_factor,
            chunks=(1, 1, 64, 128, 128),
            n_lvls=2,
        )
        array_to_write = previous_scale_pyramid[-1]

        # Create the scale dataset
        pyramid_group = zarr_group.create_dataset(
            name=i,
            shape=array_to_write.shape,
            chunks=array_to_write.chunksize,
            dtype=np.uint16,
            compressor=blosc.Blosc(cname="zstd", clevel=3, shuffle=blosc.SHUFFLE),
            dimension_separator="/",
            overwrite=True,
        )

        # Block Zarr Writer
        BlockedArrayWriter.store(array_to_write, pyramid_group, block_shape)
        written_pyramid.append(array_to_write)

    end_time = time()
    print(f"Time to write the dataset: {end_time - start_time}")
    print(f"Written pyramid: {written_pyramid}")

    try:
        client.shutdown()
    except Exception as e:
        print(f"Handling error {e} when closing client.")


def producer(
    producer_queue,
    zarr_data_loader,
    logger,
    n_consumers,
):
    """
    Function that sends blocks of data to
    the queue to be acquired by the workers.

    Parameters
    ----------
    producer_queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_data_loader: DataLoader
        Zarr data loader

    logger: logging.Logger
        Logging object

    n_consumers: int
        Number of consumers
    """
    # total_samples = sum(zarr_dataset.internal_slice_sum)
    worker_pid = os.getpid()

    logger.info(f"Starting producer queue: {worker_pid}")
    for i, sample in enumerate(zarr_data_loader):

        producer_queue.put(
            {
                'i': i,
                "data": sample.batch_tensor.numpy(),
                "batch_super_chunk": sample.batch_super_chunk[0],
                "batch_internal_slice": sample.batch_internal_slice,
            },
            block=True,
        )
        logger.info(f"[+] Worker {worker_pid} setting block {i}")

    for i in range(n_consumers):
        producer_queue.put(None, block=True)

    # zarr_dataset.lazy_data.shape
    logger.info(f"[+] Worker {worker_pid} -> Producer finished producing data.")
        
        
def consumer(
    queue,
    zarr_dataset,
    worker_params,
):
    """
    Function executed in every worker
    to acquire data.

    Parameters
    ----------
    queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_dataset: ArrayLike
        Zarr dataset

    worker_params: dict
        Worker parametes to execute a function.

    results_dict: multiprocessing.Dict
        Results dictionary where outputs
        are stored.
    """
    logger = worker_params["logger"]
    worker_results = {}
    worker_pid = os.getpid()
    logger.info(f"Starting consumer worker -> {worker_pid}")

    # Setting initial wait so all processes could be created
    # And producer can start generating data
    # sleep(60)

    # Start processing
    total_samples = sum(zarr_dataset.internal_slice_sum)

    while True:
        streamed_dict = queue.get(block=True)

        if streamed_dict is None:
            logger.info(f"[-] Worker {worker_pid} -> Turn off signal received...")
            break

        logger.info(
            f"[-] Worker {worker_pid} -> Consuming {streamed_dict['i']} - {streamed_dict['data'].shape} - Super chunk val: {zarr_dataset.curr_super_chunk_pos.value} - internal slice sum: {total_samples}"
        )

        execute_worker(
            data=streamed_dict['data'],
            batch_super_chunk=streamed_dict['batch_super_chunk'],
            batch_internal_slice=streamed_dict['batch_internal_slice'],
            cells_config=worker_params['cells_config'],
            no_cells_config=worker_params['no_cells_config'],
            overlap_prediction_chunksize=worker_params['overlap_prediction_chunksize'],
            output_destriped_zarr=worker_params['output_zarr'],
            shadow_correction=worker_params['shadow_correction'],
            dataset_name=worker_params['dataset_name'],
            logger=logger,
        )


    logger.info(f"[-] Worker {worker_pid} -> Consumer finished consuming data.")

def destripe_zarr(
    dataset_path: PathLike,
    multiscale: str,
    output_destriped_zarr: PathLike,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: Tuple[int, ...],
    results_folder: PathLike,
    derivatives_path,
    xyz_resolution,
    parameters,
    flatfield=None,
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
):
    """
    Local 3D combination of predicted gradients.
    This operation is necessary before following
    the flows to the centers identified cells.

    Parameters
    ----------
    dataset_path: str
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    output_destriped_zarr: PathLike
        Path where we want to output the destriped zarr.

    output_cellprob_path: PathLike
        Path where we want to output the cell proabability
        maps. It is not completely necessary to save them
        but it is good for quality control.

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize.

    target_size_mb: int
        Target size in megabytes the data loader will
        load in memory at a time

    n_workers: int
        Number of workers that will concurrently pull
        data from the shared super chunk in memory

    batch_size: int
        Batch size

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None

    results_folder: PathLike
        Path where the results folder for cell segmentation
        is located.

    """

    no_cells_config = parameters["no_cells_config"]
    cells_config = parameters["cells_config"]

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger = utils.create_logger(output_log_path=results_folder)
    logger.info(f"{20*'='} Large-Scale Zarr Destriping {20*'='}")

    logger.info(f"Processing dataset {dataset_path}")

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

    # Creating zarr data loader
    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    # The device we will use and pinning memory to speed things up
    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    # Getting overlap prediction chunksize
    overlap_prediction_chunksize = (
        0,
        0,
        0,
    )
    logger.info(
        f"Overlap size based on cell diameter * 2: {overlap_prediction_chunksize}"
    )

    lazy_data = (
        ImageReaderFactory()
        .create(
            data_path=dataset_path,
            parse_path=False,
            multiscale=multiscale,
        )
        .as_dask_array()
    )

    original_dataset_shape = lazy_data.shape

    logger.info(f"Lazy data shape: {lazy_data.shape}")

    # Creation of zarr data loader
    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=lazy_callback_fn,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    logger.info(f"Creating destriped zarr in path: {output_destriped_zarr}")

    store = parse_url(path=output_destriped_zarr.parent, mode="w").store
    root_group = zarr.group(store=store)
    dataset_name = Path(output_destriped_zarr).name
    new_channel_group = root_group.create_group(name=dataset_name, overwrite=True)
    output_zarr = new_channel_group.create_dataset(
        name=0,
        shape=original_dataset_shape,
        chunks=(1, 1, 64, 128, 128),
        dtype=np.uint16,
        compressor=blosc.Blosc(cname="zstd", clevel=3, shuffle=blosc.SHUFFLE),
        dimension_separator="/",
        overwrite=True,
    )

    logger.info(f"Created zarr: {output_zarr}")

    # Estimating total batches
    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(
        f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}"
    )

    logger.info(f"{20*'='} Starting combination of gradients {20*'='}")
    start_time = time()

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=exec_n_workers)

    # Variables for multiprocessing
    picked_blocks = []
    curr_picked_blocks = 0

    logger.info(f"Number of workers processing data: {exec_n_workers}")

    # Getting flatfield
    darkfield = None
    tile_config = None  # Used when the flats come from the microscope
    # If we want to apply the flats from the microscope
    apply_microscope_flats = True if flatfield is None else False
    retrospective = not apply_microscope_flats

    shading_parameters = {}

    if os.path.exists(derivatives_path):

        # Reading darkfield
        darkfield_path = str(derivatives_path.joinpath("DarkMaster_cropped.tif"))
        logger.info(f"Loading darkfield from path: {darkfield_path}")

        try:
            darkfield = tif.imread(darkfield_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Please, provide the current dark from the microscope! Provided path: {darkfield_path}"
            )

        if apply_microscope_flats:
            logger.info("Getting microscope flats!")
            channel_name = Path(output_destriped_zarr).parent.name
            # print("CHANNEL NAME: ", channel_name)
            flatfield, tile_config = get_microscope_flats(
                channel_name=str(channel_name),
                derivatives_folder=derivatives_path,
            )
            # Normalizing and inverting flatfields from the microscope
            flatfield = fl.normalize_image(flatfield)
        #             flatfield = fl.invert_image(flatfield)

        else:
            logger.info("Ignoring microscope flats...")

    if flatfield is None:
        logger.info("Estimating flats with BasicPy...")
        shading_parameters = {
            "get_darkfield": False,
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

    else:
        logging.info(
            f"Ignoring estimation of the flats from the data. Using provided flat: {flatfield.shape}"
        )

    shadow_correction = {
        "retrospective": retrospective,
        "flatfield": flatfield,  # Estimated with basicpy or using the flats from the microscope
        "darkfield": darkfield,  # Coming from the microscope
        "tile_config": tile_config,
    }
    
    #### USING QUEUES test
    
    # Create consumer processes
    factor = 20

    # Create a multiprocessing queue
    producer_queue = multiprocessing.Queue(maxsize=exec_n_workers * factor)
                    
    worker_params = {
        'cells_config': cells_config,
        'no_cells_config': no_cells_config,
        'overlap_prediction_chunksize': overlap_prediction_chunksize,
        'output_zarr': output_zarr,
        'shadow_correction': shadow_correction,
        'dataset_name': dataset_name,
        'logger': logger
    }

    logger.info(f"Setting up {exec_n_workers} workers...")
    consumers = [
        multiprocessing.Process(
            target=consumer,
            args=(
                producer_queue,
                zarr_dataset,
                worker_params,
            ),
        )
        for _ in range(exec_n_workers)
    ]

    # Start consumer processes
    for consumer_process in consumers:
        consumer_process.start()

    # Main process acts as the producer
    producer(producer_queue, zarr_data_loader, logger, exec_n_workers)
    
    # Wait for consumer processes to finish
    for consumer_process in consumers:
        consumer_process.join()
    
    end_time = time()

    scale_factor = [2, 2, 2]
    
    multiscale_time_start = time()
    
    compute_multiscale(
        output_zarr=output_zarr,
        zarr_group=new_channel_group,
        scale_factor=scale_factor,
        n_workers=co_cpus,
        voxel_size=[
            xyz_resolution[-1],
            xyz_resolution[-2],
            xyz_resolution[-3],
        ],
        image_name=dataset_name,
        n_levels=3,
        threads_per_worker=1,
    )
    multiscale_time_end = time()

    logger.info(f"Processing destripe flatfield time: {end_time - start_time} seconds")
    logger.info(f"Processing multiscale time: {multiscale_time_end - multiscale_time_start} seconds")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            results_folder,
            "zarr_destriper",
        )


def destripe_channel(
    zarr_dataset_path,
    derivatives_path,
    channel_name,
    results_folder,
    xyz_resolution,
    estimated_channel_flats,
    laser_tiles,
):
    """Main function"""
    channel_dataset = zarr_dataset_path.joinpath(channel_name)

    # Parameters for destriping
    parameters = {
        "input_path": str(channel_dataset),
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
    }
    start_time = time()

    destriped_data_folder = results_folder.joinpath("destriped_data")

    utils.create_folder(destriped_data_folder)

    for tile_path in channel_dataset.glob("*.zarr"):
        output_folder = destriped_data_folder.joinpath(f"{channel_name}/{tile_path.name}")
        print(
            f"Processing {tile_path} - writing to: {output_folder} - derivatives: {derivatives_path}"
        )

        flatfield_path = None
        for side, tiles in laser_tiles.items():
            if tile_path.stem in tiles:
                flatfield_path = estimated_channel_flats[int(side)]
                break

        if flatfield_path is None:
            raise ValueError(f"Tile {tile_path} not found in {laser_tiles}")

        flatfield = tif.imread(str(flatfield_path))
        print(f"Reading flatfield from {flatfield_path} - shape: {flatfield.shape}")

        destripe_zarr(
            dataset_path=tile_path,
            multiscale="0",
            output_destriped_zarr=output_folder,
            prediction_chunksize=(64, 1600, 2000),
            target_size_mb=3072,
            n_workers=0,
            batch_size=1,
            super_chunksize=(384, 1600, 2000),
            results_folder=results_folder,
            derivatives_path=derivatives_path,
            xyz_resolution=xyz_resolution,
            parameters=parameters,
            flatfield=flatfield,
            lazy_callback_fn=None,
        )

    end_time = time()

    generate_data_processing(
        channel_name=channel_name,
        destripe_version="0.0.1",
        destripe_config=parameters,
        start_time=start_time,
        end_time=end_time,
        output_directory=results_folder,
    )


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


def main():
    data_folder = Path(os.path.abspath("../../data"))
    results_folder = Path(os.path.abspath("../../results"))
    scratch_folder = Path(os.path.abspath("../../scratch"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/acquisition.json",
    ]

#     missing_files = validate_capsule_inputs(required_input_elements)

#     print(f"Data in folder: {list(data_folder.glob('*'))}")
    
#     if len(missing_files):
#         raise ValueError(
#             f"We miss the following files in the capsule input: {missing_files}"
#         )
    
    dask.config.set({"distributed.worker.memory.terminate": False})

    BASE_PATH = data_folder.joinpath("SmartSPIM_717381_2024-07-03_10-49-01-zarr")
    acquisition_path = data_folder.joinpath("SmartSPIM_717381_2024-07-03_10-49-01/acquisition.json")

    acquisition_dict = utils.read_json_as_dict(acquisition_path)

    if not len(acquisition_dict):
        raise ValueError(
            f"Not able to read acquisition metadata from {acquisition_path}"
        )

    voxel_resolution = get_resolution(acquisition_dict)

    derivatives_path = data_folder.joinpath("SmartSPIM_717381_2024-07-03_10-49-01/derivatives")

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
                list(data_folder.glob(f"717381_flats_test/estimated_flat_laser_{channel_name}*.tif"))
            )

            if not len(estimated_channel_flats):
                raise FileNotFoundError(
                    f"Error while retrieving flats from the data folder for channel {channel_name}"
                )

            destripe_channel(
                zarr_dataset_path=BASE_PATH,
                channel_name=channel_name,
                results_folder=results_folder,
                derivatives_path=derivatives_path,
                xyz_resolution=voxel_resolution,
                estimated_channel_flats=estimated_channel_flats,
                laser_tiles=laser_tiles,
            )

    else:
        print(f"No channels to process in {BASE_PATH}")


if __name__ == "__main__":
    main()
