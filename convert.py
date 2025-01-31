from pydub import AudioSegment
import librosa
import os

# Convert M4A to WAVytho
audio_dir = "./audio/"

for m4a_file in os.listdir(audio_dir + 'm4a'):
  audio_file = m4a_file[:-4]
  audio = AudioSegment.from_file(audio_dir +'m4a/' + m4a_file, format="m4a")
  audio.export(audio_dir + 'wav/' +  audio_file + '.wav', format="wav")
  # Now load with librosa
  output_path = audio_dir + 'wav/' + audio_file + '.wav'
  y, sr = librosa.load(output_path, sr=None)
  print(f"Loaded audio with {len(y)} samples at {sr} Hz")
