#!/bin/bash
# -----------------------------------------------
# SLURM script to derive the final RACS linguistic feature set on a directory of text files 
# ----------------------------------------------
#SBATCH -J LingPipelineGivenTranscriptions_Trial       #name of JOb
#SBATCH -o LingPipelineGivenTranscriptions_Trial.o%j   #name of std output file 
#SBATCH -e LingPipelineGivenTranscriptions_Trial.e%jn  # name of stderr output file
#SBATCH -p normal           
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu
#SBATCH --mail-type=end
CUDA_VISIBLE_DEVICES=0 
conda activate racs
#Format of command, see example below: python tacc_EnglishLingPipelineGivenTranscriptions.py inputDirectory outputName racsAWSPipelineForMADRandARTLab/data/end2end/static/LingFeatData/ 
python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_EnglishLingPipelineGivenTranscriptions.py /work/09424/smgrasso1/ls6/trialRuns/nfv_UT_transcriptions/ nfvUTTrialLingFeats /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/data/end2end/static/LingFeatData/
