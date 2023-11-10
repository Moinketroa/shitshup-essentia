FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update \
    && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libeigen3-dev libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libcublas-12-0 \
    libavutil-dev libswresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-dev python3-numpy-dev python3-numpy python3-yaml python3-six python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --ignore-installed --no-cache-dir flask

RUN pip3 install --no-cache-dir numba

RUN pip3 install --no-cache-dir essentia-tensorflow

RUN mkdir temp
RUN mkdir upload

RUN mkdir model
#download models for tensorflow opex

#ENV PYTHONPATH /usr/local/lib/python3/dist-packages

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_CACHE_MAXSIZE=2147483648

WORKDIR /essentia

EXPOSE 5000