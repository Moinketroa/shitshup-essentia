FROM ubuntu:20.04

RUN apt-get update \
    && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install tensorflow-aarch64 --no-cache-dir -f https://tf.kmtea.eu/whl/stable.html
RUN pip3 install tensorflow --no-cache-dir -f https://tf.kmtea.eu/whl/stable.html

RUN pip3 install --no-cache-dir essentia
RUN pip3 install --no-cache-dir essentia-tensorflow


#ENV PYTHONPATH /usr/local/lib/python3/dist-packages

WORKDIR /essentia