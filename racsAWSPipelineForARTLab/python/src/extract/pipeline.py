import logging
import os
import pandas as pd
import whisper
import numpy as np
import logging.config
import torch

from extract.data_dirs import DataDirs
from extract.transcribe import transcribeListOfAudioFilesOfAnyDuration
from extract.extract_features import extractFeaturesFromAudioFile
from extract.aws import extractAWSID, getLogger
from extract.enums import FileType

def configure_logging():
    """Configure the logger for the rest of the app
    """
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': { 
            'default': { 
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(logging_config)
    
    
# If it is not AWS
if not logging.getLogger(__name__).hasHandlers():
    # Configure the logging for the app
    configure_logging()
    
# Get the logger
logger = getLogger()

def run_pipeline(input_directory: str, 
                 static_files_directory: str, 
                 output_directory: str, 
                 output_file_prefix: str,
                 output_file_type: FileType = FileType.csv):
    """
    This is the main function. At this time, I'm not fully sure what it does, but it runs all code.

    Args:
        input_directory (str): The path where you expect the input files to be
        static_files_directory (str): The path to the static files
        output_directory (str): The location to write out the results
        output_file_prefix (str): The prefix for the files we are writing out
        output_file_type (FileType): The file output type
        use_s3 (bool): (Optional) Do you want to use S3? Defaults to False. If true, it will pull all directories frrm S3 instead of local
    """
    
    # Create a data directory to keep track of paths easier
    dirs: DataDirs = DataDirs(input_directory=input_directory, static_files_directory=static_files_directory, output_directory=output_directory)   
    
    logger.info("Loading Whisper model")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #logger.info("Device used is", DEVICE)
    print("Device used is", DEVICE)
    model: whisper.Whisper = whisper.load_model("large-v2", device=DEVICE) 
    logger.info("Finished loading Whisper model")
    
    # Get the list of files we want to examine
    listOfFiles: list[str] = dirs.list_input_files()   
    
    # Transcribe any files that aren't empty. Also, get a list of files that have empty transcriptions so we don't look
    # at them again
    logger.info("Transcribing files")
    transcriptions, filesToRemove = transcribeListOfAudioFilesOfAnyDuration(listOfFiles, model, dirs)
    
    # Remove the files from the "to analyze" list if we don't want to analyze them
    for file in filesToRemove:
        listOfFiles.remove(file)
        
    logger.info(f"Extracting features from {len(listOfFiles)} audio files")
    
    acousticFeats = []
    linguisticFeats = []
    
    # For every transcription
    for file_path, transcription in zip(listOfFiles, transcriptions):
        logger.debug(f"Starting feature extraction for {file_path}")
        # Get feature sets from the file
        sampleName=file_path
        fileLinguisticFeats, fileAcousticFeats = extractFeaturesFromAudioFile(file_path, sampleName, transcription, dirs)
        
        acousticFeats.append(fileAcousticFeats)
        #ignoring linguistic analysis of counting back tasks
        if 'count' not in file_path:
            linguisticFeats.append(fileLinguisticFeats) 
    
    # Get all the features in order to write them to CSV
    directoryAcousticFeats = pd.concat(acousticFeats)
    directoryLinguisticFeats = pd.concat(linguisticFeats)
    directoryAcousticFeats = directoryAcousticFeats.fillna('NA')
    directoryAcousticFeats = directoryAcousticFeats.reset_index(level=[1,2]).drop(columns=['start', 'end']) 
    directoryLinguisticFeats = directoryLinguisticFeats.fillna('NA')
    directoryLinguisticFeats = directoryLinguisticFeats.dropna()
    for col in directoryLinguisticFeats.dtypes[directoryLinguisticFeats.dtypes=='object'].index[1:]:
        directoryLinguisticFeats[col] = directoryLinguisticFeats[col].apply(lambda x: np.nan if type(x) == str and x.startswith('Skipping') else x)
        directoryLinguisticFeats[col] = directoryLinguisticFeats[col].apply(lambda x: np.nan if type(x) == str and x.startswith('Traceback') else x)
    directoryLinguisticFeats = directoryLinguisticFeats.reset_index().rename(columns={'index':'file'})
    directoryLinguisticFeats.insert(0,'AWS_ID', directoryLinguisticFeats['file'].apply(extractAWSID))
    logger.info(f"Shape of Linguistic Output: {directoryLinguisticFeats.shape}")
    
    # Write linguistics features to CSV
    linguistic_file_name: str = output_file_prefix + "_LinguisticFeatures"
    logger.info(f"Writing out linguistic features file to {linguistic_file_name}")
    
    dirs.write_to_csv(df=directoryLinguisticFeats, file_name=linguistic_file_name)
    
    # Get acoustic features ready to write to file
    directoryAcousticFeats = directoryAcousticFeats.reset_index()
    directoryAcousticFeats["file"] = directoryAcousticFeats["file"].apply(os.path.basename)
    directoryAcousticFeats.insert(0,'AWS_ID', directoryAcousticFeats['file'].apply(extractAWSID))
    logger.info(f"Shape of Acoustic Output: {directoryAcousticFeats.shape}")
    
    # Write the acoustic features to CSV
    acoustic_file_name: str = output_file_prefix + "_AcousticFeatures"
    logger.info(f"Writing out features to {acoustic_file_name}")
    
    dirs.write_to_csv(df=directoryAcousticFeats, file_name=acoustic_file_name)

    #commented out as this will only be used by ARTLab and MADR
    #if output_file_type == 'csv':
    #    dirs.write_to_csv(df=directoryAcousticFeats, file_name=acoustic_file_name) 
    #else:
    #    dirs.write_to_json(df=directoryAcousticFeats, file_name=acoustic_file_name) 
