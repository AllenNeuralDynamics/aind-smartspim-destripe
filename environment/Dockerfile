# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

# Create conda env with Python 3.9
RUN conda create -n destriper python=3.9 -y

# Use conda run as default shell inside container
SHELL ["conda", "run", "-n", "destriper", "/bin/bash", "-c"]

# Install imagecodecs + deps from conda-forge (no compilation needed)
RUN conda install -n destriper -c conda-forge imagecodecs -y

# Install Python deps via pip
RUN pip install -U --no-cache-dir \
    numpy==1.26.4 \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git@unpinned-versions \
    scipy==1.13.1 \
    scikit-image==0.24.0 \
    tifffile==2024.5.10 \
    PyWavelets==1.6.0 \
    tqdm==4.66.5 \
    pathlib2 \
    aind-data-schema==1.0.0 \
    matplotlib==3.9.2 \
    psutil==6.0.0 \
    BaSiCPy==1.1.0 \
    natsort==8.4.0 \
    imageio==2.35.1 \
    xarray_multiscale==2.1.0 \
    boto3 \
    s3fs
