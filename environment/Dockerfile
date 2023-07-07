# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n destriper python=3.8 -y
SHELL ["conda", "run", "-n", "destriper", "/bin/bash", "-c"]
# RUN conda install -c conda-forge awscli

RUN pip install -U --no-cache-dir \
    numpy==1.24.2 \
    scipy==1.9.1 \
    scikit-image==0.20.0 \
    tifffile==2023.3.21 \
    PyWavelets==1.4.1 \
    tqdm==4.65.0 \
    pathlib2==2.3.7.post1