#!/bin/bash
# -----------------------------------------------
# SLURM script to remove moderator speech
# ----------------------------------------------
#SBATCH -J removeModeratorSpeech_8f4693ac-6797-4eb5-a750-a100723d5355_treePictureRecording       #name of Job
#SBATCH -o removeModeratorSpeech_8f4693ac-6797-4eb5-a750-a100723d5355_treePictureRecording.o%j   #name of std output file 
#SBATCH -e removeModeratorSpeech_8f4693ac-6797-4eb5-a750-a100723d5355_treePictureRecording.e%jn  # name of stderr output file
#SBATCH -p normal	# 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:01:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A IRI22010

#conda activate removeModeratorSpeech
python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_removeModeratorSpeech.py 8f4693ac-6797-4eb5-a750-a100723d5355_treePictureRecording.wav