#!/bin/bash
# -----------------------------------------------
# SLURM script to say hello world
# ----------------------------------------------
#SBATCH -J Whisper_English_nfvPPA_CLAN       #name of JOb
#SBATCH -o Whisper_English_nfvPPA_CLAN.o%j   #name of std output file 
#SBATCH -e Whisper_English_nfvPPA_CLAN.e%jn  # name of stderr output file
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu#,smgrasso@austin.utexas.edu
#SBATCH --mail-type=all
#CUDA_VISIBLE_DEVICES=0 

#conda activate runWhisper
whisper --model large-v2 --language English --output_format txt --output_dir ../whisperTranscriptions_vmSmall *.wav #--device cuda 
wait

