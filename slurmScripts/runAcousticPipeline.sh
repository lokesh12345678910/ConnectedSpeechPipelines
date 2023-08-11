#!/bin/bash
# -----------------------------------------------
# SLURM script to derive the final RACS acoustic feature set on a directory of audio files 
# ----------------------------------------------
#SBATCH -J AcousticPipelineFirstRun_Spanish       #name of JOb
#SBATCH -o AcousticPipelineFirstRun_Spanish.o%j   #name of std output file 
#SBATCH -e AcousticPipelineFirstRun_Spanish.e%jn  # name of stderr output file
#SBATCH -p normal           
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu
#SBATCH --mail-type=end
CUDA_VISIBLE_DEVICES=0 
conda activate racs
#python tacc_AcousticPipelineWithTrimming.py inputDirectory outputName rVADPath 
python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_AcousticPipelineWithTrimming.py  racsTrialFiles/ trialRun /work/09424/smgrasso1/ls6/racsAWSPipeline/data/end2end/static/LingFeatData/rVAD_fast.py