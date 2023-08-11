import sys
audioFile=sys.argv[1] #"RACS_Unhealthy_June/input/"
outputPath=sys.argv[2]
rVADPath=sys.argv[3]
import numpy as np
import os
from datetime import timedelta


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
	trimmingCommand = "ffmpeg -i " + audioFile + " -ss " + startString +" -to " + lastSecondString + " -c:v  copy -c:a copy " + outputPath
	os.system(trimmingCommand)
	os.system("rm " + rVadFile)

def trimAudioFile(audioFile,outputPath,rVADPath): 
	inputFilePath = audioFile
	outputVadLabel = "RVAD_" + audioFile.split('/')[-1]
	rVADCommand = "python " + rVADPath + " " + inputFilePath + " " + outputVadLabel
	os.system(rVADCommand)
	print(rVADCommand)
	trimFileGivenRVAD(audioFile, outputVadLabel, outputPath)


trimAudioFile(audioFile, outputPath,rVADPath)
