#!/bin/bash

docker kill shitshup-essentia
docker rm shitshup-essentia

./start.sh
