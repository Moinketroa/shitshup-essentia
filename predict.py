import json
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = 'msd-musicnn-1.pb'
model_metadata_path = 'msd-musicnn-1.json'
audio_file = '/mnt/music/All Star.mp3'

musicnn_metadata = json.load(open(model_metadata_path, 'r'))

audio = MonoLoader(filename=audio_file, sampleRate=16000)()

musicnn_preds = TensorflowPredictMusiCNN(graphFilename=model_path)(audio)

print('tamer')
print(musicnn_preds)