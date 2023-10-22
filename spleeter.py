from essentia.standard import AudioLoader, TensorflowPredict, AudioWriter, MonoWriter
from essentia import Pool
import numpy as np
import os
import zipfile
import shutil

spleeter_2s_model = "models/spleeter-2s-3.pb"
spleeter_4s_model = "models/spleeter-4s-3.pb"
spleeter_5s_model = "models/spleeter-5s-3.pb"

output_directory = "/temp"
directory_2s = "2s"
directory_4s = "4s"
directory_5s = "5s"

vocals_file_name = "vocals.wav"
accompaniment_file_name = "accompaniment.wav"
drums_file_name = "drums.wav"
bass_file_name = "bass.wav"
piano_file_name = "piano.wav"
other_file_name = "other.wav"
    
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def init_spleeter_pool_input(file_name):
    audio, sr, _, _, _, _ = AudioLoader(filename=file_name)()

    pool = Pool()
    # The input needs to have 4 dimensions so that it is interpreted as an Essentia tensor.
    pool.set("waveform", audio[..., np.newaxis, np.newaxis])

    return pool, sr

def spleet_audio_2s(file_name):
    model = TensorflowPredict(
        graphFilename=spleeter_2s_model,
        inputs=["waveform"],
        outputs=["waveform_vocals", "waveform_accompaniment"]
    )

    pool, sr = init_spleeter_pool_input(file_name)

    return model(pool), sr

def spleet_audio_4s(file_name):
    model = TensorflowPredict(
        graphFilename=spleeter_4s_model,
        inputs=["waveform"],
        outputs=["waveform_vocals", "waveform_drums", "waveform_bass", "waveform_other"]
    )

    pool, sr = init_spleeter_pool_input(file_name)

    return model(pool), sr

def predict_spleeter(user_id, file_path, file_name):
    out_pool_2s, sr2 = spleet_audio_2s(file_path)
    out_pool_4s, sr4 = spleet_audio_4s(file_path)

    vocals_2s = out_pool_2s["waveform_vocals"].squeeze()
    accompaniment_2s = out_pool_2s["waveform_accompaniment"].squeeze()

    vocals_4s = out_pool_4s["waveform_vocals"].squeeze()
    drums_4s = out_pool_4s["waveform_drums"].squeeze()
    bass_4s = out_pool_4s["waveform_bass"].squeeze()
    other_4s = out_pool_4s["waveform_other"].squeeze()

    music_name, extension = os.path.splitext(file_name)

    base_directory = os.path.join(output_directory, user_id, music_name)
    base_2s_directory = os.path.join(base_directory, directory_2s)
    base_4s_directory = os.path.join(base_directory, directory_4s)
    
    create_dir(base_2s_directory)
    create_dir(base_4s_directory)

    vocals_2s_file_name = os.path.join(base_2s_directory, vocals_file_name)
    accompaniment_2s_file_name = os.path.join(base_2s_directory, accompaniment_file_name)

    vocals_4s_file_name = os.path.join(base_4s_directory, vocals_file_name)
    drums_4s_file_name = os.path.join(base_4s_directory, drums_file_name)
    bass_4s_file_name = os.path.join(base_4s_directory, bass_file_name)
    other_4s_file_name = os.path.join(base_4s_directory, other_file_name)

    print(sr2)
    print(sr4)

    AudioWriter(filename=vocals_2s_file_name)(vocals_2s)
    AudioWriter(filename=accompaniment_2s_file_name)(accompaniment_2s)

    AudioWriter(filename=vocals_4s_file_name)(vocals_4s)
    AudioWriter(filename=drums_4s_file_name)(drums_4s)
    AudioWriter(filename=bass_4s_file_name)(bass_4s)
    AudioWriter(filename=other_4s_file_name)(other_4s)

    files_to_zip = {
        vocals_2s_file_name: os.path.join(directory_2s, vocals_file_name),
        accompaniment_2s_file_name: os.path.join(directory_2s, accompaniment_file_name),

        vocals_4s_file_name: os.path.join(directory_4s, vocals_file_name),
        drums_4s_file_name: os.path.join(directory_4s, drums_file_name),
        bass_4s_file_name: os.path.join(directory_4s, bass_file_name),
        other_4s_file_name: os.path.join(directory_4s, other_file_name),
    }

    zip_file_name = music_name + '.zip'
    definitive_zip_file_name = os.path.join(output_directory, user_id, zip_file_name)

    with zipfile.ZipFile(definitive_zip_file_name, 'w') as zipf:
        for source_file, archive_path in files_to_zip.items():
            zipf.write(source_file, arcname=archive_path)

    print(f'{definitive_zip_file_name} created successfully.')     

    shutil.rmtree(base_directory)

    return definitive_zip_file_name

file_path1 = "/mnt/music/All Star.mp3"
file_name1 = "All Star.mp3"
user_id1 = "1425"

predict_spleeter(user_id1, file_path1, file_name1)