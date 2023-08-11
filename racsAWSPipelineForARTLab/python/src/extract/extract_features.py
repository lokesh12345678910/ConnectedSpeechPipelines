import pandas as pd
import spacy
import spacy_transformers
import opensmile
from typing import Any, Dict, Tuple

from extract.acoustic_feature_set import get_final_acoustic_feat_set, syllable_nuclei
from extract.data_dirs import DataDirs
from extract.linguistic_feature_helpers import filterText, setupSpacyDF
from extract.linguistic_feature_set import LinguisticFeatureSet


def extractLinguisticFeatures(transcription: str, sampleName: str, dirs: DataDirs) -> Dict[str, Any]:
    """Extract a bunch of different features from the transcription

    Args:
        transcription (str): The transcribed string
        sampleName (str): The name of the transcription

    Returns:
        dict[str, Any]: The linguistic feature set
    """
    #nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(filterText(transcription))
    df = setupSpacyDF(doc)
    
    # Holds a bunch of methods to extract the feature set
    linguisticFeatures = LinguisticFeatureSet(name=sampleName, transcription=transcription, df=df, dirs=dirs)
    
    # Add all the features we want to the dictionary
    linguisticFeatures.add_grammar_complexity()
    linguisticFeatures.add_noun_to_verb()
    linguisticFeatures.add_type_token()
    linguisticFeatures.add_moving_average_type_token(5)
    linguisticFeatures.add_moving_average_type_token(10)
    linguisticFeatures.add_moving_average_type_token(15)
    linguisticFeatures.add_moving_average_type_token(20)
    linguisticFeatures.add_propositional_density()
    linguisticFeatures.add_pos_normalized_counts()

    linguisticFeatures.add_average_sentence_length(list(doc.sents))
    
    fillers = ['okay', 'um', 'uh', 'er', 'eh']
    linguisticFeatures.add_number_fillers(fillers)
    
    linguisticFeatures.add_repetition_of_word_n(2)
    linguisticFeatures.add_splat()
    linguisticFeatures.add_word_freq()
    linguisticFeatures.add_word_length_by_phonemes()
    
    linguisticFeatures.add_semantic_diversity()
    
    linguisticFeatures.add_word_prevalence()
    linguisticFeatures.add_concreteness()
    linguisticFeatures.add_AoA()
    linguisticFeatures.add_word_length_by_morphemes()
    
    return linguisticFeatures.features

def extractAndOutputAcousticFeatures(file_path: str) -> pd.DataFrame:
    """ Get the acoustic features from a given file path
    Args:
        file_path (str): The path to the file we want to extract from

    Returns:
        DataFrame: the data frame holding the acoustic features
    """
  
    #extract features from eGeMAPS
    smile = opensmile.Smile(
        #received deprecate warning for GeMAPS w/ recommendation to use GeMAPSv01b
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    allAcousticFeatures_df: pd.DataFrame = smile.process_file(file_path)
    syllable_nuclei_dictionary: dict[str, Any] = syllable_nuclei(file_path)
    
    
    #adding syllable nuceli features to df
    for key in list(syllable_nuclei_dictionary.keys()):
        allAcousticFeatures_df[key] = syllable_nuclei_dictionary[key]
    allAcousticFeatures_df = allAcousticFeatures_df[get_final_acoustic_feat_set()]
    
    return allAcousticFeatures_df

def extractFeaturesFromAudioFile(file_name: str, 
                                 sampleName: str, 
                                 transcription: str, 
                                 dirs: DataDirs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the linguistic feature set and the acoustic feature set from a given file

    Args:
        file_name (str): The name of the file we are looking at
        sampleName (_type_): _description_
        transcription (str): The transcription we want to examine
        dirs (DataDirs): The directory to different data paths

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The linguistic feature set and the acoustic feature set
    """
    # Get the file path in relation to the input bath
    file_path = dirs.concat_with_input_path(file_name)
    
    # Get the acoustic features from the file
    acousticFeats = extractAndOutputAcousticFeatures(file_path)
    
    # Get the linguistic features from the file
    lingFeats = extractLinguisticFeatures(transcription, sampleName, dirs)

    # Convert the linguistic features to a data frame
    lingFeatsDF = pd.Series(lingFeats).to_frame().T
    lingFeatsDF.index = [sampleName]
    lingFeatsDF.rename(columns = {'Unnamed: 0': 'file'}, inplace = True)
    
    return lingFeatsDF, acousticFeats

