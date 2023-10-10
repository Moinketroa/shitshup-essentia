import json
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

model_path = 'msd-musicnn-1.pb'
model_metadata_path = 'msd-musicnn-1.json'
audio_file = '/mnt/music/All Star.mp3'

musicnn_metadata = json.load(open(model_metadata_path, 'r'))

audio = MonoLoader(filename=audio_file, sampleRate=16000)()

musicnn_preds = TensorflowPredictMusiCNN(graphFilename=model_path)(audio)

musicnn_preds = np.mean(musicnn_preds, axis=0)

musicnn_classes = musicnn_metadata['classes']

for i in range(len(musicnn_classes)):
    print('{}: {}%'.format(musicnn_classes[i], musicnn_preds[i] * 100))

print('tamer')
print(musicnn_preds)