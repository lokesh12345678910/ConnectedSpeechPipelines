import sys

inputDirectory=sys.argv[1] #"RACS_Unhealthy_June/input/"
outputName=sys.argv[2] # "RACS_Unhealthy_June_trialJan25WithSPLAT"
lingFeatDirectory = sys.argv[3] #"/work/07469/lpugalen/maverick2/LingFeatData/"
import spacy_transformers
import spacy

nlp = spacy.load('en_core_web_trf')

import nltk
nltk.download('brown')
nltk.download('names')
nltk.download('stopwords')
nltk.download('cmudict')
import math

import os
os.environ["TOKENIZERS_PARALLELISM"] =  "(true | false)"

import numpy as np


import pandas as pd



"""# Linguistic Feature Extraction Methods"""

def filterText(text):
  import nltk
  # this tokeniser takes care of contractions nicely
  from nltk.tokenize import WhitespaceTokenizer
  # Create a reference variable for Class WhitespaceTokenizer
  tk = WhitespaceTokenizer()
  trackWords = tk.tokenize(text)
  #removing unwanted characters, excluding contractions
  bad_chars = [',', ':', '!', '\"', '?']
  filteredTrackWords = [''.join(filter(lambda i: i not in bad_chars, word)) for word in trackWords]
  #remove empty words
  filteredTrackWords = [i for i in filteredTrackWords if i] 
  return " ".join(filteredTrackWords)

def setupSpacyDF(doc):
  import pandas as pd
  cols = ("text",  "POS", "STOP", "DEP")
  rows = []
  for t in doc:
    if t.pos_ == 'PUNCT' or t.pos_ == 'SYM' or t.pos_ == 'SPACE':
      #not considering punctuation
      continue
    row = [t.text, t.pos_, t.is_stop, t.dep_]
    rows.append(row)
  df = pd.DataFrame(rows, columns=cols)
  return df

def posTagCounts(df, linguisticFeatures):
  # 5.20.22: This line is problematic 
  #tag_types = list(df['POS'].unique())
  #solution: scroll to bottom of https://stackoverflow.com/questions/58215855/how-to-get-full-list-of-pos-tag-and-dep-in-spacy
  tag_types = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X", "SPACE"]
  for category in tag_types:
    tag_description = "POS_TAG:" + category
    linguisticFeatures[tag_description] = (df['POS'] == category).sum()

def posNormalizedCounts(df, linguisticFeatures):
  #tag_types = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X", "SPACE"]
  #filtered_tag_types = ["ADJ", "ADP", "ADV", "AUX","CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB"]
  #removing CONJ
  filtered_tag_types = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB"]
  #for category in tag_types:
  for category in filtered_tag_types:
    tag_description = "POS_TAG:" + category
    linguisticFeatures[tag_description] = (df['POS'] == category).sum()/df.shape[0]
  linguisticFeatures["POS_TAG:" + "CONJ"] = linguisticFeatures["POS_TAG:" + "CCONJ"] + linguisticFeatures["POS_TAG:" + "SCONJ"]

def depTagCounts(df, linguisticFeatures):
  tag_types = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]
  for category in tag_types:
    depTagDescription = "GrammarRelation:" + category
    linguisticFeatures[depTagDescription] = (df['DEP'] == category).sum()

def grammarComplexityIndex(df, linguisticFeatures):
  complexTags = ['csubj', 'csubjpass', 'ccomp', 'xcomp', 'acomp', 'pobj', 'advcl', 'mark', 'acl', 'nounmod', 'complm', 'infmod', 'partmod', 'nmod']
  numComplexTags = 0
  for tag in complexTags:
    numComplexTags += (df['DEP'] == tag).sum()
    totalNumTags = len(df['DEP'])
  grammarComplexity = numComplexTags/totalNumTags 
  linguisticFeatures['Grammar Complexity Index'] = grammarComplexity

def propositionalDensity(df, linguisticFeatures):
  pdTags = ['VERB', 'ADJ', 'ADV', 'ADP', 'CONJ', 'CCONJ', 'SCONJ']
  numPDTags = 0
  for tag in pdTags:
    numPDTags += (df['POS'] == tag).sum()
  totalNumPDTags = len(df['POS'])
  propositionalDensity = numPDTags/totalNumPDTags
  linguisticFeatures['Propositional Density'] = propositionalDensity

def typeTokenRatio(df, linguisticFeatures):
  numTokens = len(df)
  numTypes = len(set(df['text'].str.lower()))
  linguisticFeatures["# of unique words"] = numTypes
  typeTokenRatio = numTypes/numTokens
  linguisticFeatures["Type Token Ratio"] = typeTokenRatio

def mov_avg_type_token_ratio(df, window_size=50):
  sum_type_token_ratio = 0
  numWindows = 0
  window_start = 0
  window_end = window_size
  while(window_end <= len(df)):
    currentWindow = df['text'][window_start:window_end]
    numTypes = len(set(df['text'][window_start:window_end].str.lower()))
    window_type_token_ratio = numTypes/window_size
    sum_type_token_ratio += window_type_token_ratio
  #update variables for next iteration
    numWindows +=1
    window_start +=1
    window_end +=1
  if numWindows == 0:
    return pd.NA
  moving_average_type_token_ratio = sum_type_token_ratio/numWindows
  return moving_average_type_token_ratio

def determine_repetition(currentWord, words):
  for word in words:
      if (word != currentWord):
        return False
  return True

def repetition_of_word_n(n,df):
  numRepetitions = 0
  first_word = df['text'][0]
  current_word = first_word
  current_word_index = 0
  next_window_start_index = 1
  next_window_end_index = n
  next_words = None
  while(next_window_end_index < len(df)):
    nextWords = list(df['text'][next_window_start_index:next_window_end_index])
    if (determine_repetition(current_word, nextWords)):
      numRepetitions += 1
    #update for next iteration
    current_word = nextWords[0]
    current_word_index +=1
    next_window_start_index +=1
    next_window_end_index +=1
  return numRepetitions

def numFillers(df, fillers=['okay', 'um', 'uh', 'er', 'eh']):
  result = 0
  for word in list(df['text']):
    if word in fillers:
      result+=1
  return result

def averageSentenceLength(df, doc, linguisticFeatures):
  numSentences = len(list(doc.sents))
  numWords = 0
  for sent in doc.sents:
    numWords += len(sent)
  avgSentenceLength = numWords/numSentences
  linguisticFeatures["Average sentence length"] = avgSentenceLength

#ASSUMING JAVA IS INSTALLED
def splatMetrics(transcription, linguisticFeatures, fileName):
  splatOutputAddress = fileName + '_SPLAT.txt'
  with open(splatOutputAddress, 'w') as writefile:
    writefile.write(transcription)
  try:
    linguisticFeatures["yngve score"] = float(os.popen("splat yngve " + splatOutputAddress).read())
  except:
    linguisticFeatures["yngve score"] = np.nan
  try:
    linguisticFeatures["average syllables per sentence (asps)"] = float(os.popen("splat asps " + splatOutputAddress).read()) # if java is not installed, add .split("\n")[-2] after.read()
  except:
    linguisticFeatures["average syllables per sentence (asps)"] = np.nan
  try:
    linguisticFeatures["content function ratio"] = float(os.popen("splat cfr " + splatOutputAddress).read()) # if java is not installed, add .split("\n")[-2] after.read()
  except:
    linguisticFeatures["content function ratio"] = np.nan
  try:
    linguisticFeatures["Flesch-Kincaid Grade Level"] = float(os.popen("splat kincaid " + splatOutputAddress).read()) # if java is not installed, add .split("\n")[-2] after.read()
  except:
    linguisticFeatures["Flesch-Kincaid Grade Level"] = np.nan
  try:
    linguisticFeatures["Frazier Score"] = float(os.popen("splat frazier " + splatOutputAddress).read())
  except:
    linguisticFeatures["Flesch-Kincaid Grade Level"] = np.nan
  os.system("rm " + splatOutputAddress)
  os.system("rm " + splatOutputAddress + ".splat")

"""WordFrequency"""
#print(lingFeatDirectory)
textFileAddress = lingFeatDirectory + "SUBTLEXusExcel2007 - out1g.csv"
wordFreqDF = pd.read_csv(textFileAddress)
wordFreqDF['WordProb'] = wordFreqDF['FREQcount']/sum(wordFreqDF['FREQcount'])
import math 
wordFreqDF['LogWordProb'] = wordFreqDF['WordProb'].apply(math.log)
logWordProbs = dict(zip(wordFreqDF['Word'], wordFreqDF['LogWordProb']))

def computeWordFreqForWordList(wordList):
  wordFreqs = []
  smoothProb = 1/(sum(wordFreqDF['FREQcount'])+1)
  smoothLogProb = math.log(smoothProb)
  for word in wordList:
    uppercaseWord = word[0].upper() + word[1:]
    lowercaseWord = word.lower()
    wordFreq = logWordProb = np.nanmax(np.array([smoothLogProb, logWordProbs.get(lowercaseWord), logWordProbs.get(uppercaseWord)], dtype=np.float64))
    wordFreqs.append(wordFreq)
  return wordFreqs

def wordFreqMetrics(df,linguisticFeatures):
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  wordFreqList = computeWordFreqForWordList(list(openClassDF['text']))
  #print("Word Freq Vals:",wordFreqList)
  #print(np.min(wordFreqList),np.max(wordFreqList),np.mean(wordFreqList),np.std(wordFreqList))
  if len(wordFreqList) > 0:
    #linguisticFeatures["Min Word Frequency"] = np.nanmin(wordFreqList)
    #linguisticFeatures["Max Word Frequency"] = np.nanmax(wordFreqList)
    linguisticFeatures["Mean Word Frequency"] = np.nanmean(wordFreqList)
    #linguisticFeatures["SD Word Frequency"] = np.nanstd(wordFreqList)
  else:
    #linguisticFeatures["Min Word Frequency"] = np.nan
    #linguisticFeatures["Max Word Frequency"] = np.nan
    linguisticFeatures["Mean Word Frequency"] = np.nan
    #linguisticFeatures["SD Word Frequency"] = np.nan
  #return np.min(wordFreqList),np.max(wordFreqList),np.mean(wordFreqList),np.std(wordFreqList)

"""Word length by phonemes"""

phonemeDF = pd.read_csv(lingFeatDirectory + "phonemeDictionary.txt", sep="  ", names = ['Word', 'Phonemic Decomposition'])
wordPhonemeMapping = dict(zip(list(phonemeDF['Word']), list(phonemeDF['Phonemic Decomposition'])))

def computeNumPhonemesForWordList(wordList):
  wordNumPhonemes = []
  for word in wordList:
    wordPhonemes = wordPhonemeMapping.get(word.upper())
    if wordPhonemes == None:
      wordNumPhonemes.append(np.nan)
    else:
      wordNumPhonemes.append(len(wordPhonemes.split()))
  #print("# of phonemes for each word:",wordNumPhonemes)
  #print(np.min(wordNumPhonemes),np.max(wordNumPhonemes),np.mean(wordNumPhonemes),np.std(wordNumPhonemes))
  return wordNumPhonemes
  #return wordNumPhonemes.astype(np.float64)

def wordLengthByPhonemesMetrics(df,linguisticFeatures):
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  wordLengthByPhonemesList = computeNumPhonemesForWordList(list(openClassDF['text']))
  print("Word Length by Phonemes for each Word:", wordLengthByPhonemesList)
  if len(wordLengthByPhonemesList) > 0:
    #linguisticFeatures["Min Word Length By Phonemes"] = np.nanmin(wordLengthByPhonemesList)
    #linguisticFeatures["Max Word Length By Phonemes"] = np.nanmax(wordLengthByPhonemesList)
    linguisticFeatures["Mean Word Length By Phonemes"] = np.nanmean(wordLengthByPhonemesList)
    #linguisticFeatures["SD Word Length by Phonemes"] = np.nanvar(wordLengthByPhonemesList) ** 1/2
    #print("Standard Deviation of Word Length by Phonemes", linguisticFeatures["SD Word Length By Phonemes"])
  else:
    #linguisticFeatures["Min Word Length By Phonemes"] = np.nan
    #linguisticFeatures["Max Word Length By Phonemes"] = np.nan
    linguisticFeatures["Mean Word Length By Phonemes"] = np.nan
    #linguisticFeatures["SD Word Length By Phonemes"] = np.nan
  return wordLengthByPhonemesList

def computeNumPhonemesForWordList(wordList):
  wordNumPhonemes = []
  for word in wordList:
    wordPhonemes = wordPhonemeMapping.get(word.upper())
    if wordPhonemes == None:
      #wordNumPhonemes.append(np.nan)
      continue
    else:
      wordNumPhonemes.append(len(wordPhonemes.split()))
  #print("# of phonemes for each word:",wordNumPhonemes)
  #print(np.min(wordNumPhonemes),np.max(wordNumPhonemes),np.mean(wordNumPhonemes),np.std(wordNumPhonemes))
  return wordNumPhonemes
  #return wordNumPhonemes.astype(np.float64)

def wordLengthByPhonemesMetrics(df,linguisticFeatures):
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  wordLengthByPhonemesList = computeNumPhonemesForWordList(list(openClassDF['text']))
  #print("Word Length by Phonemes for each Word:", type(wordLengthByPhonemesList), wordLengthByPhonemesList)
  if len(wordLengthByPhonemesList) > 0:
    #linguisticFeatures["Min Word Length By Phonemes"] = np.nanmin(wordLengthByPhonemesList)
    #linguisticFeatures["Max Word Length By Phonemes"] = np.nanmax(wordLengthByPhonemesList)
    linguisticFeatures["Mean Word Length By Phonemes"] = np.nanmean(wordLengthByPhonemesList)
    #linguisticFeatures["SD Word Length By Phonemes"] = np.nanstd(wordLengthByPhonemesList)
    #rint("Standard Deviation of Word Length by Phonemes", linguisticFeatures["SD Word Length By Phonemes"])
  else:
    #print("Has no elements")
    #linguisticFeatures["Min Word Length By Phonemes"] = np.nan
    #print("Stored Min")
    #linguisticFeatures["Max Word Length By Phonemes"] = np.nan
    #print("Stored Max")
    linguisticFeatures["Mean Word Length By Phonemes"] = np.nan
    #print("Stored Maan")
    #linguisticFeatures["SD Word Length By Phonemes"] = np.nan
    #print("Stored SD")
 # return wordLengthByPhonemesList

"""Semantic Diversity"""

semanticDiversityDF = pd.read_csv(lingFeatDirectory + '13428_2012_278_MOESM1_ESM.csv').iloc[1:,:7]
semanticDiversityDF.rename(columns={'Supplementary Materials: SemD values': 'term', 'Unnamed: 1':'mean_cos', 'Unnamed: 2': 'SemD', 'Unnamed: 3': 'BNC_wordcount', 'Unnamed: 4': 'BNC_contexts',
                            'Unnamed: 5': 'BNC_freq', 'Unnamed: 6': 'lg_BNC_freq'},inplace=True)
semanticDiversityMapping = dict(zip(list(semanticDiversityDF['term']), list(semanticDiversityDF['SemD'])))

def computeSemanticDiversityMappingForWordList(df,linguisticFeatures):
  wordsSemanticDiversity = []
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  for word in list(openClassDF['text']):
    wordSemanticDiversity = semanticDiversityMapping.get(word.lower())
    if wordSemanticDiversity == None:
      wordsSemanticDiversity.append(np.nan)
      continue
    wordsSemanticDiversity.append(float(wordSemanticDiversity))
  if len(wordsSemanticDiversity) != 0:
    #linguisticFeatures["Min Word Semantic Diversity"] = np.nanmin(wordsSemanticDiversity)
    #linguisticFeatures["Max Word Semantic Diversity"] = np.nanmax(wordsSemanticDiversity)
    linguisticFeatures["Mean Word Semantic Diversity"] = np.nanmean(wordsSemanticDiversity)
    #linguisticFeatures["SD Word Semantic Diversity"] = np.nanstd(wordsSemanticDiversity)
  else:
    #linguisticFeatures["Min Word Semantic Diversity"] = np.nan
    #linguisticFeatures["Max Word Semantic Diversity"] = np.nan
    linguisticFeatures["Mean Word Semantic Diversity"] = np.nan
    #linguisticFeatures["SD Word Semantic Diversity"] = np.nan
  #return np.min(wordsSemanticDiversity),np.max(wordsSemanticDiversity),np.mean(wordsSemanticDiversity),np.std(wordsSemanticDiversity)

"""Age of Acquisition"""

ageOfAcquistionDF = pd.read_csv(lingFeatDirectory + 'AoA_ratings_Kuperman_et_al_BRM.csv')
ageOfAcquistionMapping = dict(zip(list(ageOfAcquistionDF['Word']), list(ageOfAcquistionDF['Rating.Mean'])))

def compute_AoA_MappingForWordList(df,linguisticFeatures):
  wordsAOAList = []
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  for word in list(openClassDF['text']):
    wordsAOA = ageOfAcquistionMapping.get(word.lower())
    if wordsAOA == None:
      wordsAOAList.append(np.nan)
      continue
    wordsAOAList.append(float(wordsAOA))
  #print("AoA for each word:",wordsAOAList)
  if len(wordsAOAList) != 0:
    #linguisticFeatures["Min Word Age of Acquisition"] = np.nanmin(wordsAOAList)
    #linguisticFeatures["Max Word Age of Acquisition"] = np.nanmax(wordsAOAList)
    linguisticFeatures["Mean Word Age of Acquisition"] = np.nanmean(wordsAOAList)
    #linguisticFeatures["SD Word Age of Acquisition"] = np.nanstd(wordsAOAList)
  else:
    #linguisticFeatures["Min Word Age of Acquisition"] = np.nan
    #linguisticFeatures["Max Word Age of Acquisition"] = np.nan
    linguisticFeatures["Mean Word Age of Acquisition"] = np.nan
    #linguisticFeatures["SD Word Age of Acquisition"] = np.nan

  #return np.min(wordsAOAList),np.max(wordsAOAList),np.mean(wordsAOAList),np.std(wordsAOAList)

"""Concreteness"""

concretenessDF = pd.read_csv(lingFeatDirectory + 'Concreteness_ratings_Brysbaert_et_al_BRM.csv')
concretenessMapping = dict(zip(list(concretenessDF['Word']), list(concretenessDF['Conc.M'])))

def compute_concreteness_MappingForWordList(df,linguisticFeatures):
  wordsConcretenessList = []
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  for word in list(openClassDF['text']):
    wordConcreteness = concretenessMapping.get(word.lower())
    if wordConcreteness == None:
      wordsConcretenessList.append(np.nan)
      continue
    wordsConcretenessList.append(float(wordConcreteness))
  #print("Concreteness for each word:",wordsConcretenessList)
  if len(wordsConcretenessList) != 0:
    #linguisticFeatures["Min Word Concreteness"] = np.nanmin(wordsConcretenessList)
    #linguisticFeatures["Max Word Concreteness"] = np.nanmax(wordsConcretenessList)
    linguisticFeatures["Mean Word Concreteness"] = np.nanmean(wordsConcretenessList)
    #linguisticFeatures["SD Word Concreteness"] = np.nanstd(wordsConcretenessList)
  else:
    #linguisticFeatures["Min Word Concreteness"] = np.nan
    #linguisticFeatures["Max Word Concreteness"] = np.nan
    linguisticFeatures["Mean Word Concreteness"] = np.nan
    #linguisticFeatures["SD Word Concreteness"] = np.nan
  #print(np.min(wordsConcretenessList),np.max(wordsConcretenessList),np.mean(wordsConcretenessList),np.std(wordsConcretenessList))
  #return np.min(wordsConcretenessList),np.max(wordsConcretenessList),np.mean(wordsConcretenessList),np.std(wordsConcretenessList)

"""Prevalence"""

prevalenceDF = pd.read_csv(lingFeatDirectory + 'WordprevalencesSupplementaryfilefirstsubmission.csv')
prevalenceMapping = dict(zip(list(prevalenceDF['Word']), list(prevalenceDF['Prevalence'])))

def compute_prevalence_MappingForWordList(df,linguisticFeatures):
  wordsPrevalenceList = []
  openClassPOSTags = ['NOUN', 'VERB', 'ADJ', 'ADV']
  openClassDF = df[df.POS.isin(openClassPOSTags)]
  for word in list(openClassDF['text']):
    wordPrevalence = prevalenceMapping.get(word.lower())
    if wordPrevalence == None:
      wordsPrevalenceList.append(np.nan)
      continue
    wordsPrevalenceList.append(float(wordPrevalence))
  #print("Prevalence for each word:",wordsPrevalenceList)
  if len(wordsPrevalenceList) != 0:
    #linguisticFeatures["Min Word Prevalence"] = np.nanmin(wordsPrevalenceList)
    #linguisticFeatures["Max Word Prevalence"] = np.nanmax(wordsPrevalenceList)
    linguisticFeatures["Mean Word Prevalence"] = np.nanmean(wordsPrevalenceList)
    #linguisticFeatures["SD Word Prevalence"] = np.nanstd(wordsPrevalenceList)
  else:
    #linguisticFeatures["Min Word Prevalence"] = np.nan
    #linguisticFeatures["Max Word Prevalence"] = np.nan
    linguisticFeatures["Mean Word Prevalence"] = np.nan
    #linguisticFeatures["SD Word Prevalence"] = np.nan
  #return np.min(wordsPrevalenceList),np.max(wordsPrevalenceList),np.mean(wordsPrevalenceList),np.std(wordsPrevalenceList)

"""# General Methods"""

def extractLinguisticFeatures(transcription, sampleName):
  doc = nlp(filterText(transcription))
  df = setupSpacyDF(doc)
  linguisticFeatures = {}
  grammarComplexityIndex(df, linguisticFeatures)
  linguisticFeatures['Noun to Verb ratio'] = (df['POS'] == 'NOUN').sum()/(df['POS'] == 'VERB').sum() 
  #linguisticFeatures['# of unique words per word'] = len(set(df['text'].str.lower()))/df.shape[0]
  #This is by def, type token ratio
  typeTokenRatio(df, linguisticFeatures)
  linguisticFeatures["Moving Average Type Token Ratio, n=5"] = mov_avg_type_token_ratio(df, window_size=5)
  linguisticFeatures["Moving Average Type Token Ratio, n=10"] = mov_avg_type_token_ratio(df, window_size=10)
  linguisticFeatures["Moving Average Type Token Ratio, n=15"] = mov_avg_type_token_ratio(df, window_size=15)
  linguisticFeatures["Moving Average Type Token Ratio, n=15"] = mov_avg_type_token_ratio(df, window_size=20)
  propositionalDensity(df, linguisticFeatures)
  posNormalizedCounts(df, linguisticFeatures)
  linguisticFeatures["Average sentence length"] = len(transcription.split())/len(list(doc.sents))
  fillers = ['okay', 'ok', 'um', 'uh', 'er', 'eh']
  linguisticFeatures["# of Fillers per word"] = numFillers(df, fillers)/df.shape[0]
  linguisticFeatures["Repetitions per word"] = repetition_of_word_n(2,df)/df.shape[0]
  splatMetrics(transcription, linguisticFeatures, sampleName)
  wordFreqMetrics(df,linguisticFeatures)
  wordLengthByPhonemesMetrics(df,linguisticFeatures)
  computeSemanticDiversityMappingForWordList(df,linguisticFeatures)
  compute_prevalence_MappingForWordList(df,linguisticFeatures)
  compute_concreteness_MappingForWordList(df,linguisticFeatures)
  compute_AoA_MappingForWordList(df,linguisticFeatures)
  #wordLengthByMorphemes(df,linguisticFeatures)
  #computeAffectiveRatingsForWordList(df,linguisticFeatures) 63 feats
  return linguisticFeatures


def extractFeaturesFromFile(sampleName, transcription):
  lingFeats = extractLinguisticFeatures(transcription, sampleName)
  lingFeatsDF = pd.Series(lingFeats).to_frame().T
  lingFeatsDF.index = [sampleName]
  lingFeatsDF.rename(columns = {'Unnamed: 0': 'file'}, inplace = True)
  return lingFeatsDF


def extractAWSIDForLinguistic(sampleName):
  return sampleName.split('_')[0]

def getLingFeatsFromDirectory(path):
    linguisticFeats = []
    for file in sorted(os.listdir(path)):
        if file.endswith('.txt'):
            if 'count' not in file:
                transcription = open(path + file).read()
                fileLinguisticFeats = extractFeaturesFromFile(file,transcription.lower())
                linguisticFeats.append(fileLinguisticFeats)
    return linguisticFeats

def runPipeline(inputDirectory, fileName):
  assert inputDirectory[-1] == '/', "Last character of input path must be /"
  import os
  linguisticFeats  = getLingFeatsFromDirectory(inputDirectory)
  import pandas as pd
  directoryLinguisticFeats = pd.concat(linguisticFeats)
  directoryLinguisticFeats = directoryLinguisticFeats.fillna('NA')
  directoryLinguisticFeats = directoryLinguisticFeats.dropna()
  for col in directoryLinguisticFeats.dtypes[directoryLinguisticFeats.dtypes=='object'].index[1:]:
    directoryLinguisticFeats[col] = directoryLinguisticFeats[col].apply(lambda x: np.nan if type(x) == str and x.startswith('Skipping') else x)
    directoryLinguisticFeats[col] = directoryLinguisticFeats[col].apply(lambda x: np.nan if type(x) == str and x.startswith('Traceback') else x)
  directoryLinguisticFeats = directoryLinguisticFeats.reset_index().rename(columns={'index':'file'})
  directoryLinguisticFeats.insert(0,'AWS_ID', directoryLinguisticFeats['file'].apply(extractAWSIDForLinguistic))
  print("Shape of Linguistic Output:", directoryLinguisticFeats.shape)
  directoryLinguisticFeats.to_csv(fileName + "_LinguisticFeatures.csv", index=False)




runPipeline(inputDirectory, outputName)
