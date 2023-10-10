docker run -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music myimage


docker run -ti --rm -v `pwd`:/essentia -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music mtgupf/essentia python3 test.py


docker run -it -v `pwd`:/essentia -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music mtgupf/essentia



docker run -it -v `pwd`:/essentia -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music mtgupf/essentia-tensorflow

pip uninstall -y tensorflow-cpu && pip install -U https://tf.novaal.de/barcelona/tensorflow-2.3.4-cp36-cp36m-linux_x86_64.whl && python3 -c "import tensorflow as tf; tf.print(\"hello world\")"

https://tf.novaal.de/barcelona/tensorflow-2.3.4-cp36-cp36m-linux_x86_64.whl





docker build -t shitshup-essentia:1.0 .





docker run -it -m 8192m --memory-reservation=4096m --memory-swap="1g" -v `pwd`:/essentia -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music shitshup-essentia:1.0


docker run -it -v `pwd`:/essentia -v /Users/jimeme/Work/temp/youtube-dl/test/:/mnt/music shitshup-essentia:1.0

winpty docker run -it --gpus=all -v /`pwd -W`://essentia -v "/C:\Users\MasterChief\Work\Shitshup\ressources\mediathèque"://mnt/music shitshup-essentia:1.0





curl -SLO https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb
curl -SLO https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.json


winpty docker run -ti --gpus=all -v /`pwd -W`://essentia -v "/C:\Users\MasterChief\Work\Shitshup\ressources\mediathèque"://mnt/music shitshup-essentia:1.0 python3 music.py



docker build -t shitshup-essentia:1.0 .
winpty docker run -d --name shitshup-essentia --gpus=all -v /`pwd -W`://essentia -v "/C:\Users\MasterChief\Work\Shitshup\ressources\mediathèque"://mnt/music shitshup-essentia:1.0 python3 shitshup-essentia-server.py