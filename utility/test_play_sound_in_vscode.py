import os

from pydub import AudioSegment
from pydub.playback import play

file_path = "./audio/wav/wangbaochuan-1.wav"

audio = AudioSegment.from_wav(file_path)
play(audio)

print(os.path.exists(file_path))


# If you're using an IDE like VS Code, you won’t get native audio support, 
# but you can test playback outside Jupyter using standard libraries 
# like pygame or pydub

