# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n destriper python=3.9 -y
SHELL ["conda", "run", "-n", "destriper", "/bin/bash", "-c"]
# RUN conda install -c conda-forge awscli

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    numpy \
    scipy \
    scikit-image \
    tifffile \
    PyWavelets \
    tqdm \
    pathlib2 \
<<<<<<< HEAD
    aind-data-schema==1.0.0 \
=======
    aind-data-schema==0.38.6 \
>>>>>>> 82b3f804642574371b9dcaf20d76b268c1772f32
    matplotlib \
    psutil \
    BaSiCPy==1.1.0 \
    natsort \
    git+https://github.com/imageio/imageio.git \
    xarray_multiscale==2.1.0