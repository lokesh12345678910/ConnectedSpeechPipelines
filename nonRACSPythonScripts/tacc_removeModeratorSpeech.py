import sys
import os
audioFile=sys.argv[1]

monoAudioFile = "mono_" + audioFile.split('.')[0] +  '.wav'
os.system("ffmpeg -i " + audioFile+ " -ac 1 " + monoAudioFile)
audioFile = monoAudioFile 

##ffmpeg -i audioFile -ac 1 'mono'+audioFile+'.mp3'
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",  use_auth_token="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")

diarization = pipeline(audioFile)

print("When the moderator speaks, assuming moderator speaks less than participant")
for turn, _, speaker in diarization.itertracks(yield_label=True):
  #if speaker != 'SPEAKER_00':
  print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

data =[]
for turn, _, speaker in diarization.itertracks(yield_label=True):
  data.append([turn.start,turn.end,speaker])

import pandas as pd
df = pd.DataFrame(data, columns = ['start','stop','speaker'])
from collections import Counter
speechToRemoveDF = df[df.speaker != Counter(df['speaker']).most_common(1)[0][0]] #assuming moderator speaks less than participant

speaker0DF = df[df.speaker == 'SPEAKER_00']
speaker1DF = df[df.speaker == 'SPEAKER_01']
print(len(speaker0DF),len(speaker1DF))
import numpy as np

import pydub
from pydub import AudioSegment

def remove_time_range(input_file, output_file, start_time, end_time):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the start and end positions in milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    #print(start_ms, end_ms)
    # Split the audio at the specified time range
    first_part = audio[:start_ms]
    second_part = audio[end_ms:]

    # Concatenate the two parts
    result = first_part + second_part

    # Export the resulting audio to a new file
    result.export(output_file, format="wav")


#remove speech from speaker0 to form speaker1 audio file
speaker0TimeStamps = list(zip(speaker0DF['start'],speaker0DF['stop']))
speaker0TimeStamps.reverse()
#print(speaker0TimeStamps)
for start, stop in speaker0TimeStamps:
  if os.path.exists("trimmed_Speaker1_" + audioFile):
    remove_time_range("trimmed_Speaker1_" + audioFile, "trimmed_Speaker1_" + audioFile, start, stop)
  else:
    remove_time_range(audioFile, "trimmed_Speaker1_" + audioFile, start, stop)



#remove speech from speaker1 to form speaker0 audio file

speaker1TimeStamps = list(zip(speaker1DF['start'],speaker1DF['stop']))
speaker1TimeStamps.reverse() #reversing is very important to ensure removing from end of audio file, not start
#print(speaker1TimeStamps)
for start, stop in speaker1TimeStamps:
  if os.path.exists("trimmed_Speaker0_" + audioFile):
    remove_time_range("trimmed_Speaker0_" + audioFile, "trimmed_Speaker0_" + audioFile, start, stop)
  else:
    remove_time_range(audioFile, "trimmed_Speaker0_"+ audioFile, start, stop)