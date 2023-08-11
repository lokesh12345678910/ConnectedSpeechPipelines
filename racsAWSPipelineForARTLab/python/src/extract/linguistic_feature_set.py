import pandas as pd
import numpy as np
from typing import Any, List
import math
import os
from splat.SPLAT import SPLAT

from polyglot.text import Text, Word #for word length by morphemes
from polyglot.downloader import downloader
downloader.download("morph2.en") #terminal alternative: polyglot download morph2.en

from extract.data_dirs import DataDirs

class LinguisticFeatureSet:
    """A bunch of methods that let us get features out of the data set
    """
    def __init__(self, 
                 name: str,
                 df: pd.DataFrame, 
                 transcription: str, 
                 dirs: DataDirs):
        """
        Args:
            name (str): Name of the transcription
            df (pd.DataFrame): Dataframe for the feature
            transcription (str): Transcription for the feature
            dirs (DataDirs): The data directory
        """
        self.name = name
        self.df: pd.DataFrame = df
        self.transcription: str = transcription
        self.dirs: DataDirs = dirs
        # This holds the features 
        self.features: dict[str, Any] = {}
        
    def add_grammar_complexity(self) -> None:
        """
        Modify the dictionary and add the grammar complexity index.
        """
        complexTags = ['csubj', 'csubjpass', 'ccomp', 'xcomp', 'acomp', 'pobj', 'advcl', 'mark', 'acl', 'nounmod', 'complm', 'infmod', 'partmod', 'nmod']
        numComplexTags = 0
        for tag in complexTags:
            numComplexTags += (self.df['DEP'] == tag).sum()
            totalNumTags = len(self.df['DEP'])
        grammarComplexity = numComplexTags/totalNumTags 
        self.features['Grammar Complexity Index'] = grammarComplexity
        
    def add_noun_to_verb(self) -> None:
        self.features['Noun to Verb ratio'] = (self.df['POS'] == 'NOUN').sum()/(self.df['POS'] == 'VERB').sum() 
        
    def add_type_token(self) -> None:
        """Add type token ratio and number of unique words to the dictionary
        """
        #This is by def, type token ratio
        numTokens = len(self.df)
        numTypes = len(set(self.df['text'].str.lower()))
        self.features["# of unique words"] = numTypes
        typeTokenRatio = numTypes/numTokens
        self.features["Type Token Ratio"] = typeTokenRatio
        
    def add_moving_average_type_token(self, window_size: int = 50) -> None:
        """
        Add metrics for moving type token size
        Args:
            window_size (int, optional)
        """
        key_name = f"Moving Average Type Token Ratio, n={window_size}"
        sum_type_token_ratio = 0
        numWindows = 0
        window_start = 0
        window_end = window_size
        while(window_end <= len(self.df)):
            currentWindow = self.df['text'][window_start:window_end]
            numTypes = len(set(self.df['text'][window_start:window_end].str.lower()))
            window_type_token_ratio = numTypes/window_size
            sum_type_token_ratio += window_type_token_ratio
            #update variables for next iteration
            numWindows +=1
            window_start +=1
            window_end +=1
        if numWindows == 0:
            return pd.NA
        
        moving_average_type_token_ratio = sum_type_token_ratio/numWindows
        
        self.features[key_name] = moving_average_type_token_ratio

    def add_propositional_density(self) -> None:
        """Add propritional density to the dictionary
        """
        pdTags = ['VERB', 'ADJ', 'ADV', 'ADP', 'CONJ', 'CCONJ', 'SCONJ']
        numPDTags = 0
        for tag in pdTags:
            numPDTags += (self.df['POS'] == tag).sum()
        totalNumPDTags = len(self.df['POS'])
        propositionalDensity = numPDTags/totalNumPDTags
        self.features['Propositional Density'] = propositionalDensity
        
    def add_pos_normalized_counts(self) -> None:
        """Add POS_TAG info to the dictionary
        """
        filtered_tag_types = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB"]
        for category in filtered_tag_types:
            tag_description = "POS_TAG:" + category
            self.features[tag_description] = (self.df['POS'] == category).sum()/self.df.shape[0]
        self.features["POS_TAG:" + "CONJ"] = self.features["POS_TAG:" + "CCONJ"] + self.features["POS_TAG:" + "SCONJ"]

    def add_number_fillers(self, fillers: List[str]) -> None:
        """Add the number of fillers that are in the transcription

        Args:
            fillers (List[str]): The filler words you want to use
        """
        num_fillers = 0
        for word in list(self.df['text']):
            if word in fillers:
                num_fillers+=1
                
        self.features["# of Fillers per word"] = num_fillers/self.df.shape[0]
        
    def add_repetition_of_word_n(self, n: int) -> None:
        """Add metric for repetitions per word
        Args:
            n (int)
        """
        def determine_repetition(currentWord, words):
            for word in words:
                if (word != currentWord):
                    return False
            return True
        
        numRepetitions = 0
        first_word = self.df['text'][0]
        current_word = first_word
        current_word_index = 0
        next_window_start_index = 1
        next_window_end_index = n
        next_words = None
        while(next_window_end_index < len(self.df)):
            nextWords = list(self.df['text'][next_window_start_index:next_window_end_index])

            if (determine_repetition(current_word, nextWords)):
                numRepetitions += 1
            #update for next iteration
            current_word = nextWords[0]
            current_word_index +=1
            next_window_start_index +=1
            next_window_end_index +=1
            
        repetition_per_word = numRepetitions/self.df.shape[0]
        self.features["Repetitions per word"] = repetition_per_word
        
    def _computeWordFreqForWordList(self, wordList: List[str]) -> List:
        # Read in data from the appropriate data directory
        # This used to be outside of the function so I'm just moving it around
        logWordProbs = self.dirs.log_word_frequencey_mapping()
        wordFreqs = []
        smoothLogProb = math.log(self.dirs.word_freq_smooth_prob())
        for word in wordList:
            uppercaseWord = word[0].upper() + word[1:]
            lowercaseWord = word.lower()
            wordFreq = logWordProb = np.nanmax(np.array([smoothLogProb, logWordProbs.get(lowercaseWord), logWordProbs.get(uppercaseWord)], dtype=np.float64))
            wordFreqs.append(wordFreq)
        return wordFreqs
        
    def add_word_freq(self) -> None:
        """Add word frequency metrics to the feature set.
        """
        
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        wordFreqList = self._computeWordFreqForWordList(list(openClassDF['text']))
        if len(wordFreqList) > 0:
            self.features["Mean Word Frequency"] = np.nanmean(wordFreqList)
        else:
            self.features["Mean Word Frequency"] = np.nan
            
    def _computeNumPhonemesForWordList(self, wordList: List[str]) -> List:
        wordNumPhonemes = []
        for word in wordList:
            wordPhonemes = self.dirs.phoneme_mapping().get(word.upper())
            if wordPhonemes == None:
                # TODO: should this be continue? There are two methods with the same name
                wordNumPhonemes.append(np.nan)
            else:
                wordNumPhonemes.append(len(wordPhonemes.split()))
        return wordNumPhonemes

            
    def add_word_length_by_phonemes(self) -> None:
        """Add length of phonemes metrics to the feature set
        """
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        wordLengthByPhonemesList = self._computeNumPhonemesForWordList(list(openClassDF['text']))
        if len(wordLengthByPhonemesList) > 0:
            self.features["Mean Word Length By Phonemes"] = np.nanmean(wordLengthByPhonemesList)
        else:
            self.features["Mean Word Length By Phonemes"] = np.nan
            
            
    def add_semantic_diversity(self) -> None:
        """Add metrics for semantic diversity
        """
        wordsSemanticDiversity = []
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        for word in list(openClassDF['text']):
            wordSemanticDiversity = self.dirs.semantic_diversity_mapping().get(word.lower())
            if wordSemanticDiversity == None:
                wordsSemanticDiversity.append(np.nan)
                continue
            wordsSemanticDiversity.append(float(wordSemanticDiversity))
        if len(wordsSemanticDiversity) != 0:
            self.features["Mean Word Semantic Diversity"] = np.nanmean(wordsSemanticDiversity)
        else:
            self.features["Mean Word Semantic Diversity"] = np.nan    
            
    def add_word_prevalence(self) -> None:
        """Add word prevalence metric to features
        """
        wordsPrevalenceList = []
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        for word in list(openClassDF['text']):
            wordPrevalence = self.dirs.word_prevalence_mapping().get(word.lower())
            if wordPrevalence == None:
                wordsPrevalenceList.append(np.nan)
                continue
            wordsPrevalenceList.append(float(wordPrevalence))
        if len(wordsPrevalenceList) != 0:
            self.features["Mean Word Prevalence"] = np.nanmean(wordsPrevalenceList)
        else:
            self.features["Mean Word Prevalence"] = np.nan
            
    def add_word_length_by_morphemes(self) -> None:
        """Add word length by morphemes metric to features
        """
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        wordsLengthByMophemes = [len(Word(word, language="en").morphemes) for word in list(openClassDF['text'])]
        try: 
            self.features["Mean Word Length By Morphemes"] = np.mean(wordsLengthByMophemes)
        except:
            self.features["Mean Word Length By Morphemes"] = np.nan
            
    def add_concreteness(self) -> None:
        """Add concreteness metric to features
        """
        # Get the data from the file using the data dirs paths 
        wordsConcretenessList = []
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        for word in list(openClassDF['text']):
            wordConcreteness = self.dirs.concreteness_mapping().get(word.lower())
            if wordConcreteness == None:
                wordsConcretenessList.append(np.nan)
                continue
            wordsConcretenessList.append(float(wordConcreteness))
        if len(wordsConcretenessList) != 0:
            self.features["Mean Word Concreteness"] = np.nanmean(wordsConcretenessList)
        else:
            self.features["Mean Word Concreteness"] = np.nan
            
    def add_AoA(self) -> None:
        """Add age of acquisition metric to features
        """
        # Get data and read it in based on directory structure
        wordsAOAList = []
        openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
        openClassDF = self.df[self.df.POS.isin(openClassPOSTags)]
        for word in list(openClassDF['text']):
            wordsAOA = self.dirs.age_of_acquisition_mapping().get(word.lower())
            if wordsAOA == None:
                wordsAOAList.append(np.nan)
                continue
            wordsAOAList.append(float(wordsAOA))
            
        if len(wordsAOAList) != 0:
            self.features["Mean Word Age of Acquisition"] = np.nanmean(wordsAOAList)
        else:
            self.features["Mean Word Age of Acquisition"] = np.nan   
            
    def add_splat(self) -> None:
        """Add splat metrics. ASSUMING JAVA IS INSTALLED
        """
        # Create SPLAT object like they do here: https://github.com/meyersbs/SPLAT/blob/master/splat/splat
        # This is better than running external commands via os
        splat = SPLAT(self.transcription)
        
        try:
            self.features["yngve score"] = float(splat.tree_based_yngve_score())
        except:
            self.features["yngve score"] = np.nan
        try:
            self.features["average syllables per sentence (asps)"] = float(splat.average_sps()) # if java is not installed, add .split("\n")[-2] after.read()
        except:
            self.features["average syllables per sentence (asps)"] = np.nan
        try:
            self.features["content function ratio"] = float(splat.content_function_ratio()) # if java is installed, add .split("\n")[-2] after.read()
        except:
            self.features["content function ratio"] = np.nan
        try:
            self.features["Flesch-Kincaid Grade Level"] = float(splat.kincaid_grade_level()) # if java is installed, add .split("\n")[-2] after.read()
        except:
            self.features["Flesch-Kincaid Grade Level"] = np.nan
        try:
            self.features["Frazier Score"] = float(splat.tree_based_frazier_score())
        except:
            self.features["Frazier Score"] = np.nan
        
    def add_average_sentence_length(self, doc_sents: list) -> None:
        """Add metrics for average sentence length
        """
        self.features["Average sentence length"] = len(self.transcription.split())/len(list(doc_sents))
        
        