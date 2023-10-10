#!/bin/bash

docker build -t shitshup-essentia:1.0 .
winpty docker run -d -p 5000:5000 --name shitshup-essentia --gpus=all -v /`pwd -W`://essentia -v "/C:\Users\MasterChief\Work\Shitshup\ressources\mediathèque"://mnt/music shitshup-essentia:1.0 python3 shitshup-essentia-server.py

/git-bash.exe -c 'docker logs -f shitshup-essentia && exec $SHELL' & \
/git-bash.exe -c 'winpty docker exec -it shitshup-essentia bash && exec $SHELL' &