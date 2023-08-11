#!/bin/bash
# -----------------------------------------------
# SLURM script to run the RACS AWS pipeline on a directory
# ----------------------------------------------
#SBATCH -J racsAWSPipelineStephanieTrialCSV       #name of JOb
#SBATCH -o racsAWSPipelineStephanieTrialCSV.o%j   #name of std output file 
#SBATCH -e racsAWSPipelineStephanieTrialCSV.e%jn  # name of stderr output file
#SBATCH -p normal           
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:15:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu
#SBATCH --mail-type=end
CUDA_VISIBLE_DEVICES=0 
conda init bash
conda activate racs
python -m spacy download en_core_web_trf
python -m nltk.downloader brown stopwords names cmudict
python /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/python/src/app.py --input-dir ./racsTrialFiles --output-dir ./racsSlurmTrialCSVOutput/ --static-dir /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/data/end2end/static/LingFeatData/ --output-file-prefix trial --output-file-type csv
wait