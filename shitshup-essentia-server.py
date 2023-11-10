from flask import Flask, request, json, jsonify, send_file
from essentia.standard import MusicExtractor, YamlOutput, AudioLoader, MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredictVGGish, TensorflowPredict2D, TensorflowPredict, AudioWriter
from essentia import Pool
from numba import njit, cuda
import numpy as np
import tempfile
import os
import zipfile
import shutil

app = Flask(__name__)

effnet_extractor_path = 'models/discogs-effnet-bs64-1.pb'
vggish_extractor_path = 'models/audioset-vggish-3.pb'

model_sigmoid_graph_node_name = 'model/Sigmoid'
model_softmax_graph_node_name = 'model/Softmax'

jamendo_genre_model_path = 'models/mtg_jamendo_genre-discogs-effnet-1.pb'
jamendo_genre_metadata_path = 'models/mtg_jamendo_genre-discogs-effnet-1.json'

approachability_model_path = 'models/approachability_2c-discogs-effnet-1.pb'
approachability_metadata_path = 'models/approachability_2c-discogs-effnet-1.json'

engagement_model_path = 'models/engagement_2c-discogs-effnet-1.pb'
engagement_metadata_path = 'models/engagement_2c-discogs-effnet-1.json'

timbre_model_path = 'models/timbre-discogs-effnet-1.pb'
timbre_metadata_path = 'models/timbre-discogs-effnet-1.json'

danceability_model_path = 'models/danceability-audioset-vggish-1.pb'
danceability_metadata_path = 'models/danceability-audioset-vggish-1.json'

aggresive_model_path = 'models/mood_aggressive-audioset-vggish-1.pb'
aggresive_metadata_path = 'models/mood_aggressive-audioset-vggish-1.json'

happy_model_path = 'models/mood_happy-audioset-vggish-1.pb'
happy_metadata_path = 'models/mood_happy-audioset-vggish-1.json'

party_model_path = 'models/mood_party-audioset-vggish-1.pb'
party_metadata_path = 'models/mood_party-audioset-vggish-1.json'

relaxed_model_path = 'models/mood_relaxed-audioset-vggish-1.pb'
relaxed_metadata_path = 'models/mood_relaxed-audioset-vggish-1.json'

sad_model_path = 'models/mood_sad-audioset-vggish-1.pb'
sad_metadata_path = 'models/mood_sad-audioset-vggish-1.json'

spleeter_2s_model = "models/spleeter-2s-3.pb"
spleeter_4s_model = "models/spleeter-4s-3.pb"
spleeter_5s_model = "models/spleeter-5s-3.pb"

output_directory = "/temp"
directory_2s = "2s"
directory_4s = "4s"

vocals_file_name = "vocals.wav"
accompaniment_file_name = "accompaniment.wav"
drums_file_name = "drums.wav"
bass_file_name = "bass.wav"
other_file_name = "other.wav"

def predict_music_data(audio_signal, model_path, metadata_path, graph_node_name, embedding_model, data, prediction_category):
    embeddings = embedding_model(audio_signal)

    model = TensorflowPredict2D(graphFilename=model_path, output=graph_node_name)
    predictions = model(embeddings)
    predictions = np.mean(predictions, axis=0)

    metadata = json.load(open(metadata_path, 'r'))
    classes = metadata['classes']

    data[prediction_category] = {}

    for i in range(len(classes)):
        data[prediction_category][classes[i]] = predictions[i].astype(float)

def predict_music_data_effnet_extractor(audio_signal, model_path, metadata_path, graph_node_name, data, prediction_category):
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=effnet_extractor_path, output="PartitionedCall:1")
    predict_music_data(audio_signal, model_path, metadata_path, graph_node_name, embedding_model, data, prediction_category)

def predict_music_data_vggish_extractor(audio_signal, model_path, metadata_path, graph_node_name, data, prediction_category):
    embedding_model = TensorflowPredictVGGish(graphFilename=vggish_extractor_path, output="model/vggish/embeddings")
    predict_music_data(audio_signal, model_path, metadata_path, graph_node_name, embedding_model, data, prediction_category)

def music_data_predictions(file_path):
    audio_signal = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    predictions_data = {}

    predict_music_data_effnet_extractor(audio_signal, jamendo_genre_model_path, jamendo_genre_metadata_path, model_sigmoid_graph_node_name, predictions_data, 'genre')
    predict_music_data_effnet_extractor(audio_signal, approachability_model_path, approachability_metadata_path, model_softmax_graph_node_name, predictions_data, 'approachability')
    predict_music_data_effnet_extractor(audio_signal, engagement_model_path, engagement_metadata_path, model_softmax_graph_node_name, predictions_data, 'engagement')
    predict_music_data_effnet_extractor(audio_signal, timbre_model_path, timbre_metadata_path, model_softmax_graph_node_name, predictions_data, 'timbre')

    predict_music_data_vggish_extractor(audio_signal, danceability_model_path, danceability_metadata_path, model_softmax_graph_node_name, predictions_data, 'danceability')
    predict_music_data_vggish_extractor(audio_signal, aggresive_model_path, aggresive_metadata_path, model_softmax_graph_node_name, predictions_data, 'aggresive')
    predict_music_data_vggish_extractor(audio_signal, happy_model_path, happy_metadata_path, model_softmax_graph_node_name, predictions_data, 'happy')
    predict_music_data_vggish_extractor(audio_signal, party_model_path, party_metadata_path, model_softmax_graph_node_name, predictions_data, 'party')
    predict_music_data_vggish_extractor(audio_signal, relaxed_model_path, relaxed_metadata_path, model_softmax_graph_node_name, predictions_data, 'relaxed')
    predict_music_data_vggish_extractor(audio_signal, sad_model_path, sad_metadata_path, model_softmax_graph_node_name, predictions_data, 'sad')

    return predictions_data

    
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def init_spleeter_pool_input(file_name):
    audio, sample_rate, _, _, _, _ = AudioLoader(filename=file_name)()

    pool = Pool()
    pool.set("waveform", audio[..., np.newaxis, np.newaxis])

    return pool, sample_rate

def spleet_audio_2s(file_name):
    model = TensorflowPredict(
        graphFilename=spleeter_2s_model,
        inputs=["waveform"],
        outputs=["waveform_vocals", "waveform_accompaniment"]
    )

    pool, sample_rate = init_spleeter_pool_input(file_name)

    return model(pool), sample_rate

def spleet_audio_4s(file_name):
    model = TensorflowPredict(
        graphFilename=spleeter_4s_model,
        inputs=["waveform"],
        outputs=["waveform_vocals", "waveform_drums", "waveform_bass", "waveform_other"]
    )

    pool, sample_rate = init_spleeter_pool_input(file_name)

    return model(pool), sample_rate

def predict_spleeter(user_id, file_path, file_name):
    out_pool_2s, sample_rate_2s = spleet_audio_2s(file_path)
    out_pool_4s, sample_rate_4s = spleet_audio_4s(file_path)

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

    AudioWriter(filename=vocals_2s_file_name, sampleRate=sample_rate_2s)(vocals_2s)
    AudioWriter(filename=accompaniment_2s_file_name, sampleRate=sample_rate_2s)(accompaniment_2s)

    AudioWriter(filename=vocals_4s_file_name, sampleRate=sample_rate_4s)(vocals_4s)
    AudioWriter(filename=drums_4s_file_name, sampleRate=sample_rate_4s)(drums_4s)
    AudioWriter(filename=bass_4s_file_name, sampleRate=sample_rate_4s)(bass_4s)
    AudioWriter(filename=other_4s_file_name, sampleRate=sample_rate_4s)(other_4s)

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

def music_data_standard(file_path):
    features, features_frames = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                              rhythmStats=['mean', 'stdev'],
                                              tonalStats=['mean', 'stdev'])(file_path)

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name

        yaml_output = YamlOutput(filename=temp_file_path, format='json')
        yaml_output(features)

    with open(temp_file_path, 'r') as file:
        standard_data = file.read()

    os.remove(temp_file_path)

    parsed_standard_data = json.loads(standard_data)
    return parsed_standard_data




@app.route('/musicData/<userId>', methods=['POST'])
def post_music_data(userId):
    try:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file to a location
            encoded_file_name = uploaded_file.filename
            upload_directory = "/upload/" + userId

            if not os.path.exists(upload_directory):
                os.mkdir(upload_directory)

            file_path = os.path.join(upload_directory, encoded_file_name)
            uploaded_file.save(file_path)

            standard_data = music_data_standard(file_path)
            predictions_data = music_data_predictions(file_path)
            data = {}
            data['standard'] = standard_data
            data['predictions'] = predictions_data

            os.remove(file_path)
            return jsonify(data)
        else:
            return jsonify({'error': 'No file selected'})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/spleeter/<userId>', methods=['POST'])
def post_spleeter(userId):
    try:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file to a location
            encoded_file_name = uploaded_file.filename
            upload_directory = "/upload/" + userId

            if not os.path.exists(upload_directory):
                os.mkdir(upload_directory)

            file_path = os.path.join(upload_directory, encoded_file_name)
            uploaded_file.save(file_path)

            attachment_file_path = predict_spleeter(userId, file_path, encoded_file_name)

            os.remove(file_path)
            return send_file(attachment_file_path, as_attachment=True)
        else:
            return jsonify({'error': 'No file selected'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)