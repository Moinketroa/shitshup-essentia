FROM ubuntu:20.04

RUN apt-get update \
    && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install flask

RUN pip3 install --no-cache-dir tensorflow

RUN pip3 install --no-cache-dir essentia
RUN pip3 install --no-cache-dir essentia-tensorflow

RUN mkdir temp
RUN mkdir upload

RUN mkdir model
#download models for tensorflow opex

#ENV PYTHONPATH /usr/local/lib/python3/dist-packages

WORKDIR /essentia

EXPOSE 5000