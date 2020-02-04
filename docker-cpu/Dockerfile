FROM ubuntu:18.04

# Basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install HDF5 packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libhdf5-serial-dev \
    h5utils \
    hdf5-tools && \
    rm -rf /var/lib/apt/lists/*

# Install openssh (Is this needed?)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openssh-server \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PYTHON_VERSION=3.7

RUN curl -v -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

# Install ismrmrd-python and ismrmrd-python-tools
RUN conda install -c anaconda h5py && \
    conda install -c conda-forge pyxb && \
    mkdir -p /opt/code && \
    cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python.git && \
    cd ismrmrd-python && \
    pip install . && \
    cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd ismrmrd-python-tools && \
    pip install .

RUN pip install sigpy && \
    conda install -c conda-forge matplotlib && \
    conda install -c conda-forge scikit-image && \
    conda install -c anaconda pandas && \
    conda install -c anaconda scikit-learn && \
    cd /opt/code && \
    git clone https://github.com/chcodes/mricoils.git

# TODO: Make git repos pip installable
ENV PYTHONPATH "/opt/code/pc-mri:/opt/code/mricoils:${PYTHONPATH}"

# Requires my pc-mri to be available locally
COPY pc-mri /opt/code/pc-mri

RUN cd /opt/code && \
    git clone https://github.com/chcodes/python-ismrmrd-server.git
