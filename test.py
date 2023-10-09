from essentia.standard import *
# from essentia.standard import MonoLoader, Duration, Danceability, Key, PercivalBpmEstimator, RhythmDescriptors

# import MonoLoader, Duration, Danceability from essentia.standard

# Load an audio file
audio_file = '/mnt/music/All Star.mp3'
loader = MonoLoader(filename=audio_file)  # Load as mono audio
audio = loader()

#################### DURATION ###########################
# Calculate the duration of the audio in seconds
duration = Duration()

print(f"Duration: {duration(audio)} seconds")

#################### DANCEABILITY #######################
# Calculate the danceability of the audio (between 0 and 3)
danceability = Danceability()
d_danceability, d_dfa = danceability(audio)

print(f"Danceability: {d_danceability}")

####################### KEY #############################
# Calculate the key of the audio
frameCutter = FrameCutter()
spectrum = Spectrum()
spectralPeaks = SpectralPeaks()
hpcp = HPCP()
key = Key()

f_frame = frameCutter(audio)
s_spectrum = spectrum(f_frame)
sp_frequencies, sp_magnitudes = spectralPeaks(s_spectrum)
h_hpcp = hpcp(sp_frequencies, sp_magnitudes)

k_key, k_scale, k_strength, k_firstToSecondRelativeStrength = key(h_hpcp)

print(f"Key: {k_key}")
print(f"Key scale: {k_scale}")
print(f"Key estimation strenght: {k_strength}")

####################### BPM #############################
# Calculate the BPM of the audio
bpm = PercivalBpmEstimator()
b_bpm = bpm(audio)

print(f"Percival BPM: {b_bpm}")

# Calculate the BPM (other method)
descriptor = RhythmDescriptors()
descriptor_output = descriptor(audio)

rd_confidence = descriptor_output[1]
rd_bpm = descriptor_output[2]

print(f"RhythmDescriptors BPM: {rd_bpm}")
print(f"RhythmDescriptors Confidence: {rd_confidence}")

