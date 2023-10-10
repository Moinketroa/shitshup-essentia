from flask import Flask, request, json, jsonify
from essentia.standard import MusicExtractor, YamlOutput
import tempfile
import os


app = Flask(__name__)


def music_data(file_path):
    features, features_frames = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                              rhythmStats=['mean', 'stdev'],
                                              tonalStats=['mean', 'stdev'])(file_path)

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name

        yaml_output = YamlOutput(filename=temp_file_path, format='json')
        yaml_output(features)

    with open(temp_file_path, 'r') as file:
        json_data = file.read()

    os.remove(temp_file_path)

    return json_data




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

            #result = subprocess.check_output(['./music-data.sh', file_path], shell=True)
            #data = json.loads(result)
            data = music_data(file_path)
            json_data = json.loads(data)
            return jsonify(json_data)
        else:
            return jsonify({'error': 'No file selected'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)