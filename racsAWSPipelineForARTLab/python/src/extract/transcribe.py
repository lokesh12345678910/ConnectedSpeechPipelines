from typing import Any, List, Tuple
import logging
import whisper
import os
from extract.data_dirs import DataDirs
from extract.aws import getLogger
import numpy as np
from datetime import timedelta

logger = getLogger()

def transcribeListOfAudioFilesOfAnyDuration(audioFileList: List[str], 
                                            whisper_model: whisper.Whisper, 
                                            dirs: DataDirs) -> Tuple[List[Any], List[str]]:
    """
    Transcribe all files in the list. Split them into a list of transcribed files and a list of files that have empty transcriptions that 
    we don't want to keep.
    
    Args:
        audioFileList: The list of files we want to examine
        whisper_model: The whisper model already loaded (small.en)
        dirs: The data directory that holds directory info
        
    Returns:
        a tuple holding the transcriptions and the files that hold empty transcriptions
    """
    # Get all input files
    transcriptions: List[str] = []
    filesToRemove: List[str] = []

    print("Static File directory:", dirs._static_files_directory)
    os.system("mkdir wavFiles")
    os.system("mkdir monoWavFiles")
    os.system("mkdir trimmedMonoWavFiles")
    for audioFile in audioFileList:
        # Get the full audio path file
        
        
        audio_file_path = dirs.concat_with_input_path(audioFile)
        
        logger.info(f"Making audio file a wav file")
        wavAudioFileName = audioFile.replace(audioFile[-3:], "wav") #dealing with. oog input
        assert wavAudioFileName[-4:] == ".wav"
        os.system("ffmpeg -i " + audio_file_path + " " + os.path.join('wavFiles',wavAudioFileName))
  
        logger.info(f"Making audio file mono")
        os.system("ffmpeg -i " + os.path.join('wavFiles',wavAudioFileName) + " -ac 1 " + "monoWavFiles/mono_" + wavAudioFileName)
        
        logger.info("Trimming audio file")
        rVADPath = os.path.join(dirs._static_files_directory,'rVAD_fast.py')
        trimAudioFile(wavAudioFileName,'trimmedMonoWavFiles/trimmed_' + wavAudioFileName,rVADPath) 
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        transcription = whisper_model.transcribe("trimmedMonoWavFiles/trimmed_" + wavAudioFileName)["text"]       
        # If a transcription isn't empty, keep the transcription
        if transcription != '' and len(transcription.split()) != 0:
            transcriptions.append(transcription.lower())
        # If a transcription is empty, list it as a file that we want to remove
        else:
            filesToRemove.append(audioFile)
            
    os.system("rm -r wavFiles")
    os.system("rm -r monoWavFiles")
    os.system("rm -r trimmedMonoWavFiles")
    return transcriptions, filesToRemove    

def trimAudioFile(audioFile,outputPath,rVADPath):     
    monoFilePath = "monoWavFiles/mono_" + audioFile
    outputVadPath = "trimmedMonoWavFiles/" + "RVAD_" + audioFile
    rVADCommand = "python " + rVADPath + " " + monoFilePath + " " + outputVadPath 
    os.system(rVADCommand)
   #print(rVADCommand)
    trimFileGivenRVAD(monoFilePath, outputVadPath, outputPath)

def trimFileGivenRVAD(audioFile, rVadFile, outputPath):
    f = open(rVadFile)
    rVADOutput = f.read()
    rVADOutput = rVADOutput.split('\n')
    rVADOutput = [int(ele) for ele in rVADOutput[:-1]]
    startTimeFrame = rVADOutput.index(1)
    lastTimeFrame = len(rVADOutput) - rVADOutput[::-1].index(1) - 1
    startSecond = int(np.floor(startTimeFrame /100))
    startMilisecond = int(np.floor((startTimeFrame % 100)/100 * 60))
    lastSecond = int(np.floor(lastTimeFrame /100))
    lastMilisecond = int(np.floor((lastTimeFrame % 100)/100 * 60))
    lastSecondString = str(timedelta(seconds=lastSecond,milliseconds=lastMilisecond))
    startString = str(timedelta(seconds=startSecond,milliseconds=startMilisecond))
    #Trying to replicate this command
    trimmingCommand = "ffmpeg -i " + audioFile + " -ss " + startString +" -to " + lastSecondString + " -c:v  copy -c:a copy " + outputPath
    os.system(trimmingCommand)
    os.system("rm " + rVadFile)