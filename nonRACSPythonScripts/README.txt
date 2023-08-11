Running RACS pipeline:
python /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/python/src/app.py --input-dir ./racsTrialFiles --static-dir /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/data/end2end/static/LingFeatData/ --output-file-prefix trial --output-file-type csv
-----------------------------------------------------------------------------------------------
Running Eng Ling Feat pipeline that takes in transcriptions:
General Format: python tacc_EnglishLingPipelineGivenTranscriptions.py inputDirectory outputName lingFeatDirectory
Running on Stephanie's TACC: python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_EnglishLingPipelineGivenTranscriptions.py /work/09424/smgrasso1/ls6/trialRuns/nfv_UT_transcriptions/ nfvUTTrialLingFeats /work/09424/smgrasso1/ls6/racsAWSPipelineForMADRandARTLab/data/end2end/static/LingFeatData/
------------------------------------------------------------------------------------------------------------
Running Acoustic Feat Pipeline:
General Format: python tacc_AcousticPipelineWithTrimming.py inputDirectory outputName rVADPath
python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_AcousticPipelineWithTrimming.py  racsTrialFiles/ trialRun /work/09424/smgrasso1/ls6/racsAWSPipeline/data/end2end/static/LingFeatData/rVAD_fast.py
------------------------------------------------------------------------------------------------------------
Removing moderator speech:
python /work/09424/smgrasso1/ls6/nonRACSPythonScripts/tacc_removeModeratorSpeech.py 8f4693ac-6797-4eb5-a750-a100723d5355_treePictureRecording.wav
------------------------------------------------------------------------------------------------------------------------------------------------------
Run whisper for all the wav files in current directory:
whisper --model large-v2 --language English --output_dir racsTrialSlurmOutput --output_format txt *.wav
--------------------------------------------------------------------------------------------------------------------
Run Spanish Ling Feat pipeline:
conda activate spanishPipeline
python spanishLingPipeline.py inputDirectory outputName lingFeatDirectory
e.g. python spanishLingPipeline.py /work/09424/smgrasso1/ls6/trialRuns/spanishPipelineTrialFiles/ spanishPipelineTrialOutput /work/09424/smgrasso1/ls6/SpanishLingFeatData/
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Run Catalan Ling Feat pipeline:
conda activate spanishPipeline
python catalanLingPipeline.py inputDirectory outputName lingFeatDirectory
e.g. python catalanLingPipeline.py /work/09424/smgrasso1/ls6/CatalanPipelineRuns/trialFiles/ CatalanTrialPipelineOutput /work/09424/smgrasso1/ls6/CatalanLingFeatData/





