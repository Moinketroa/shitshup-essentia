from essentia.standard import *
from tempfile import TemporaryDirectory
import subprocess

audio_file = '/mnt/music/All Star.mp3'

features, features_frames = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                              rhythmStats=['mean', 'stdev'],
                                              tonalStats=['mean', 'stdev'])(audio_file)

temp_dir = TemporaryDirectory()
results_file = temp_dir.name + '/results.json'

YamlOutput(filename=results_file, format="json")(features)

# Preview the resulting file.
try:
    # Use the 'cat' command to display the contents of the temporary file
    subprocess.run(['cat', results_file], check=True, text=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")