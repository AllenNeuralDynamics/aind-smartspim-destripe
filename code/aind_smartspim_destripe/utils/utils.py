"""
Utility functions
"""

import json
import logging
import multiprocessing
import os
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import psutil
from natsort import natsorted


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def create_logger(output_log_path: str) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/destripe_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    # Trying to get CPU cores from Code Ocean
    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1

    # Trying to get CPU cores from SLURM
    slurm_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")

    # Total cpus in node SLURM_CPUS_ON_NODE
    if slurm_cpus:
        return slurm_cpus

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def get_memory_limit_bytes():
    """
    Gets the best estimate of the memory limit (in bytes) for the current job.
    Order of precedence:
    1. CO_MEMORY environment variable (assumed in GB)
    2. Cgroup memory limit (from /sys/fs/cgroup/)
    3. SLURM environment variables
    4. psutil system memory (total)
    """
    # 1. CO_MEMORY (in GB)
    memory_env = os.environ.get("CO_MEMORY")
    if memory_env:
        try:
            return int(memory_env)  # Convert GB → bytes
        except ValueError:
            pass  # Invalid format, fallback

    # 2. cgroup memory limit (in bytes)
    cgroup_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    try:
        with open(cgroup_path, "r") as f:
            mem_bytes = int(f.read().strip())
            # Some systems report a huge number when no limit is set
            if mem_bytes < 1 << 50:  # Filter out values >1PB
                return mem_bytes
    except FileNotFoundError:
        pass

    # 3. SLURM memory allocation
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")  # in MB
    if mem_per_node:
        try:
            return int(mem_per_node) * 1024**2  # MB → bytes
        except ValueError:
            pass

    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")  # in MB
    cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if mem_per_cpu and cpus:
        try:
            return int(mem_per_cpu) * int(cpus) * 1024**2  # MB → bytes
        except ValueError:
            pass

    # 4. Fallback: system-wide total memory
    return psutil.virtual_memory().total


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    memory = get_memory_limit_bytes()

    if memory:
        memory = int(memory)
        memory = get_size(memory)

    slurm_id = os.environ.get("SLURM_JOBID")
    # System info
    sep = "=" * 20
    logger.info(f"{sep} Machine Information {sep}")
    logger.info(f"Assigned cores: {get_cpu_limit()}")
    logger.info(f"Assigned memory: {memory} GBs")
    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")
    logger.info(f"Is pipeline execution in SLURM?: {bool(slurm_id)}")
    logger.info(f"SLURM ID: {slurm_id}")
    logger.info(f"SLURM GPUs: {os.environ.get('SLURM_JOB_GPUS')}")
    logger.info(f"SLURM CPUs: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")
    logger.info(
        f"SLURM variables {[( k, v ) for k, v in os.environ.items() if 'SLURM' in k]}"
    )

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def read_image_directory_structure(folder_dir: str, channel_regex: str) -> dict:
    """
    Creates a dictionary representation of all the images
    saved by folder/col_N/row_N/images_N.[file_extention]

    Parameters
    ------------------------
    folder_dir:PathLike
        Path to the folder where the images are stored

    channel_regex: str
        Regular expression to match the folders

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = natsorted(
        [
            folder_dir.joinpath(folder)
            for folder in os.listdir(folder_dir)
            if os.path.isdir(folder_dir.joinpath(folder))
            and re.search(channel_regex, str(folder))
        ]
    )

    if not len(channel_paths):
        raise ValueError(f"No channels found in path: {folder_dir}")

    cols = natsorted(os.listdir(channel_paths[0]))
    column_example = channel_paths[0].joinpath(cols[0])
    rows = natsorted(os.listdir(column_example))
    images = natsorted(os.listdir(column_example.joinpath(rows[0])))

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}

        for col in cols:
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}

                for row in rows:
                    possible_row = (
                        channel_paths[channel_idx].joinpath(col).joinpath(row)
                    )

                    if os.path.isdir(possible_row):
                        directory_structure[channel_paths[channel_idx]][col][
                            row
                        ] = images

    return directory_structure


def create_folder(dest_dir: str, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


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


def list_s3_folders(bucket: str, prefix: str, extension: Optional[str] = None) -> list:
    """
    List top-level 'folders' under a given S3 prefix that end with a given extension.

    Parameters
    ----------
        bucket: str
            Name of the S3 bucket.
        prefix: str
            S3 prefix path (e.g., "my/path/"), must end with "/".
        extension: str
            Extension to match folder names against (e.g., ".tif", ".zip").

    Returns
    -------
        list: A list of matching folder prefixes (strings ending with "/").
    """
    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    folders = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            folder_name = Path(cp["Prefix"].rstrip("/")).name
            if extension is None or folder_name.endswith(extension):
                folders.append(folder_name)

    return folders


def list_s3_files(bucket: str, prefix: str, extension: str) -> list:
    """
    List files under a given S3 prefix that end with a given extension.

    Parameters
    ----------
    bucket: str
        Name of the S3 bucket.
    prefix: str
        S3 prefix path (e.g., "my/path/"), must end with "/".
    extension: str
        Extension to match file names against (e.g., ".tif", ".zip").

    Returns
    -------
    list: A list of matching file keys.
    """
    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(extension):
                files.append(key)

    return files


def is_s3_path(path: str) -> bool:
    """
    Checks if a path is an s3 path

    Parameters
    ----------
    path: str
        Provided path

    Returns
    -------
    bool
        True if it is a S3 path,
        False if not.
    """
    parsed = urlparse(str(path))
    return parsed.scheme == "s3"


def split_s3_path(s3_path: str):
    """
    Split an S3 URI into bucket and prefix.

    Parameters
    ----------
    s3_path : str
        Example: "s3://my-bucket/folder1/folder2/"

    Returns
    -------
    (bucket, prefix) : tuple[str, str]
    """
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    # remove leading slash
    prefix = parsed.path.lstrip("/")
    return bucket, prefix
