import os
import pandas as pd
import math
from typing import List

from extract.aws import getLogger

logger = getLogger()

class DataDirs:
    """This holds general data directory info like directory paths and it holds files that we are reading in repeatedly.
    This makes it easier to mass change locations. For example, if you want to use it locally, you can create a DataDirs based around your TACC folder location. If we are on AWS,
    it allows us to easily create directories around S3 file locations.
    """
    def __init__(self, input_directory: str, static_files_directory: str, output_directory: str):
        """
        Args:
            input_directory (str): Directory where the input data is. This should hold data that needs to be processed such as acoustic files
            static_files_directory (str): This should hold our static data that we reuse, such as phoneme files
        """
        # EX: "RACS_Unhealthy_June/input/"
        logger.info("Creating data dirs")
        logger.info(f"Input directory: {input_directory}")
        self.input_directory: str = input_directory
        # EX: "/work/07469/lpugalen/maverick2/LingFeatData/"
        logger.info(f"Static files directory: {static_files_directory}")
        self._static_files_directory: str = static_files_directory
        
        logger.info(f"Output directory: {output_directory}")
        self.output_directory: str = output_directory
        
        # Some directories that we lazily set
        # Do not call these directly!!! call them through their associated method
        self._semantic_diversity_mapping = None
        self._concreteness_mapping = None
        self._phoneme_mapping = None
        self._age_of_acquisition_mapping = None
        self._log_word_freq_mapping = None
        self._word_freq_smooth_prob = None
        self._word_prevalence_mapping = None
        
    def semantic_diversity_mapping(self) -> dict:
        """Get the semantic diversity mapping from the file 13428_2012_278_MOESM1_ESM.csv
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: dictionary of contents
        """
        # Semantic diversity external file
        if self._semantic_diversity_mapping is None:
            semanticDiversityFile: str = self._append_to_static_files_directory('13428_2012_278_MOESM1_ESM.csv')
            logger.info(f"Reading in the semantic diversity file from {semanticDiversityFile}")
            semanticDiversityDF = pd.read_csv(semanticDiversityFile).iloc[1:,:7]
            semanticDiversityDF.rename(columns={'Supplementary Materials: SemD values': 'term', 'Unnamed: 1':'mean_cos', 'Unnamed: 2': 'SemD', 'Unnamed: 3': 'BNC_wordcount', 'Unnamed: 4': 'BNC_contexts',
                                'Unnamed: 5': 'BNC_freq', 'Unnamed: 6': 'lg_BNC_freq'},inplace=True)
            self._semantic_diversity_mapping: dict = dict(zip(list(semanticDiversityDF['term']), list(semanticDiversityDF['SemD'])))
        
        return self._semantic_diversity_mapping
    
    def concreteness_mapping(self) -> dict:
        """Get the concreteness mapping from the following file Concreteness_ratings_Brysbaert_et_al_BRM.csv.
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: contents of file
        """
        if self._concreteness_mapping is None:
            concretenessFile: str = self._append_to_static_files_directory('Concreteness_ratings_Brysbaert_et_al_BRM.csv')
            logger.info(f"Reading in the concreteness file from {concretenessFile}")
            concretenessDF = pd.read_csv(concretenessFile)
            self._concreteness_mapping = dict(zip(list(concretenessDF['Word']), list(concretenessDF['Conc.M'])))   

        return self._concreteness_mapping
        
    def phoneme_mapping(self) -> dict:
        """Get the phoneme mapping from the following file phonemeDictionary.txt
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: contents of the file
        """
        # phoneme dictionary external file
        if self._phoneme_mapping is None:
            phonemeFile: str = self._append_to_static_files_directory('phonemeDictionary.txt')
            logger.info(f"Reading in the phoneme file from {phonemeFile}")
            phonemeDF = pd.read_csv(phonemeFile, sep="  ", names = ['Word', 'Phonemic Decomposition'])
            self._phoneme_mapping: dict = dict(zip(list(phonemeDF['Word']), list(phonemeDF['Phonemic Decomposition'])))
        
        return self._phoneme_mapping
    
    def word_prevalence_mapping(self) -> dict:
        """Get the word prevalence mapping from the following file WordprevalencesSupplementaryfilefirstsubmission.csv
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: the contents of the file
        """
        # Word prevalence dictionary external file
        if self._word_prevalence_mapping is None:
            wordPrevalenceFile: str = self._append_to_static_files_directory('WordprevalencesSupplementaryfilefirstsubmission.csv')
            logger.info(f"Reading in the word prevalence file from {wordPrevalenceFile}")
            wordPrevalenceDf = pd.read_csv(wordPrevalenceFile)
            self._word_prevalence_mapping: dict = dict(zip(list(wordPrevalenceDf['Word']), list(wordPrevalenceDf['Prevalence'])))
            
        return self._word_prevalence_mapping
    
    def age_of_acquisition_mapping(self) -> dict:
        """Get the age of acquisition mapping from the following file AoA_ratings_Kuperman_et_al_BRM.csv
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: contents of the file
        """
        if self._age_of_acquisition_mapping is None:
            # Age of acquisition
            ageOfAcquisitionFile: str = self._append_to_static_files_directory('AoA_ratings_Kuperman_et_al_BRM.csv')
            logger.info(f"Reading in the age of acquisition file from {ageOfAcquisitionFile}")
            ageOfAcquistionDF = pd.read_csv(ageOfAcquisitionFile)
            self._age_of_acquisition_mapping = dict(zip(list(ageOfAcquistionDF['Word']), list(ageOfAcquistionDF['Rating.Mean'])))
        
        return self._age_of_acquisition_mapping

    def log_word_frequencey_mapping(self) -> dict:
        """Get the log word frequency mapping from the following file SUBTLEXusExcel2007 - out1g.csv and then apply
        a log to word prob column
        Note: I'm caching it so we don't open call it until we need it

        Returns:
            dict: contents of the file
        """
        if self._log_word_freq_mapping is None:
            self._calc_word_frequency()
        return self._log_word_freq_mapping
        
    def word_freq_smooth_prob(self) -> float:
        """Get the word frequency smooth probability from aggregations of the following file: SUBTLEXusExcel2007 - out1g.csv
        Note: I'm caching it so we don't open call it until we need it
        
        Returns:
            float: the word frequency smooth probability
        """
        if self._word_freq_smooth_prob is None:
            self._calc_word_frequency()
            
        return self._word_freq_smooth_prob
        
    def _calc_word_frequency(self) -> None:
        """Calculate the log word frequency mapping and aggregations from it
        """
        # Word frequency external file
        wordFrequencyFile: str = self._append_to_static_files_directory('SUBTLEXusExcel2007 - out1g.csv')
        logger.info(f"Reading in the word frequency file from {wordFrequencyFile}")
        wordFreqDF = pd.read_csv(wordFrequencyFile)
        wordFreqDF['WordProb'] = wordFreqDF['FREQcount']/sum(wordFreqDF['FREQcount'])
        wordFreqDF['LogWordProb'] = wordFreqDF['WordProb'].apply(math.log)
        self._word_freq_smooth_prob = 1/(sum(wordFreqDF['FREQcount']) + 1)
        self._log_word_freq_mapping = dict(zip(wordFreqDF['Word'], wordFreqDF['LogWordProb'])) 
        
        
    def _append_to_static_files_directory(self, file_name: str) -> str:
        """Append a file to the working directory

        Args:
            file_name (str): The file nname you want appended

        Returns:
            str: The full file contents
        """
        return os.path.join(self._static_files_directory, file_name)
    
    def list_input_files(self) -> List[str]:
        """List all files in the input directory

        Returns:
            List[str]: A list of the input files in the input directory
        """
        listOfFiles = sorted(os.listdir(self.input_directory))
        if '.ipynb_checkpoints' in listOfFiles:
            listOfFiles.remove('.ipynb_checkpoints')
            
        return listOfFiles
    
    def concat_with_input_path(self, file_name: str) -> str:
        """Concatenate a file name with the input path

        Args:
            file_name (str): The file name you wanted added to the path

        Returns:
            str: The path to the file
        """
        return os.path.join(self.input_directory, file_name)
        
    def write_to_json(self, df: pd.DataFrame, file_name: str) -> None:
        assert not file_name.endswith(".json"), "Do not put the file extension in the name"
        """Write a data frame to JSON

        Args:
            df (pd.DataFrame): The data frame you want added
            file_name (str): The file name. This shouldn't be a path
        """
        file_path = os.path.join(self.output_directory, f"{file_name}.json")
        df.to_json(file_path, orient='records', indent=4)
        
    def write_to_csv(self, df: pd.DataFrame, file_name: str) -> None:
        assert not file_name.endswith(".csv"), "Do not put the file extension in the name"
        """Write a data frame to CSV

        Args:
            df (pd.DataFrame): The data frame you want added
            file_name (str): The file name. This shouldn't be a path
        """
        file_path = os.path.join(self.output_directory, f"{file_name}.csv")
        df.to_csv(file_path, index=False)