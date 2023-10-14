from flask import Flask, request, json, jsonify
from essentia.standard import MusicExtractor, YamlOutput, MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredictVGGish, TensorflowPredict2D
import tempfile
import os
import numpy as np

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




@app.route('/musicData', methods=['POST'])
def post_music_data():
    try:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file to a location
            encoded_file_name = uploaded_file.filename
            upload_directory = "/upload/"
            file_path = os.path.join(upload_directory, encoded_file_name)
            uploaded_file.save(file_path)

            standard_data = music_data_standard(file_path)
            predictions_data = music_data_predictions(file_path)
            data = {}
            data['standard'] = standard_data
            data['predictions'] = predictions_data

            return jsonify(data)
        else:
            return jsonify({'error': 'No file selected'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)