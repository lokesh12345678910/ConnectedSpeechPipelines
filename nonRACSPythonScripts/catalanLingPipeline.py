import sys

inputDirectory=sys.argv[1] #/ "work/09424/smgrasso1/ls6/CatalanPipelineRuns/trialFiles/
outputName=sys.argv[2] # "CatalanTrialPipelineOutput"
lingFeatDirectory ="/work/09424/smgrasso1/ls6/CatalanLingFeatData/"

import spacy_transformers
import spacy

nlp = spacy.load('ca_core_news_trf')
nlp = nlp.from_disk(lingFeatDirectory + "catalanSpaCyModel_June16")
import math
import pandas as pd
import numpy as np
import nltk
catalanLingFeatDirectory = lingFeatDirectory 
import epitran

import os
os.environ["TOKENIZERS_PARALLELISM"] =  "(true | false)"




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
  #doc = nlp(text)
  import pandas as pd
  cols = ("text",  "POS", "STOP", "DEP", "MORPH")
  rows = []
  for t in doc:
    if t.pos_ == 'PUNCT' or t.pos_ == 'SYM' or t.pos_ == 'SPACE':
      continue
    if t.text == 'caminando':
      rows.append([t.text, "VERB", t.is_stop, t.dep_, t.morph])
      continue
    if t.text.isupper():
      rows.append([t.text, "NOUN", t.is_stop, t.dep_, t.morph])
      continue
    if t.pos_== 'PROPN' and t.text.islower():
      rows.append([t.text, "NOUN", t.is_stop, t.dep_, t.morph])
      continue
    if t.text == ',':
      #resolves special case for BILP10_PreTx_Catalan_WABPicnic
      continue
    if t.text == 'a':
      rows.append([t.text, "ADP", t.is_stop, t.dep_, t.morph])
      continue
    if t.text == 'al':
      rows.append(['a', "ADP", np.nan,np.nan,np.nan])
      rows.append(['l', "DET", np.nan,np.nan, np.nan])
      continue
    if t.text == 'seixanta':
      rows.append([t.text, "NUM", t.is_stop, t.dep_, t.morph])
      continue
    if t.text == 'dels':
      rows.append(['de', "ADP", np.nan,np.nan,np.nan])
      rows.append(['els', "DET", np.nan,np.nan, np.nan])
      continue
    row = [t.text, t.pos_, t.is_stop, t.dep_, t.morph]
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

def depNormalizedTagCounts(df, linguisticFeatures):
  tag_types = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]
  for category in tag_types:
    depTagDescription = "DEP_Proportion:" + category
    linguisticFeatures[depTagDescription] = (df['DEP'] == category).sum()/df.shape[0]


def verbFormNormalizedCounts(df, linguisticFeatures):
  verbFormTypes = ['Fin', 'Ger', 'Inf', 'Part']#, 'NA']
  verbFormCounts = {}
  for verbForm in verbFormTypes:
    verbFormCounts[verbForm] = 0
  for wordMorph in df['MORPH']:
    wordMorph = str(wordMorph)
    if 'VerbForm=' not in wordMorph:
      continue
    verbForm = wordMorph.split('VerbForm=')[1]
    assert verbForm in verbFormTypes, print(verbForm)
    verbFormCounts[verbForm] +=1
  #normalization
  numVerbForms = sum(verbFormCounts.values())
  for key in verbFormCounts.keys():
    verbFormCounts[key] /= numVerbForms
  #assert list(verbFormCounts.keys()) == verbFormTypes
  for category in verbFormTypes:
    verbFormDescription = "VerbForm_Proportion:" + category
    linguisticFeatures[verbFormDescription] = verbFormCounts[category]
  return linguisticFeatures


def moodNormalizedCounts(df, linguisticFeatures):
  moodTypes=['Ind', 'Imp', 'Sub', 'Cnd']
  moodCounts = {}
  for mood in moodTypes:
    moodCounts[mood] = 0
  for wordMorph in df['MORPH']:
    wordMorph = str(wordMorph)
    #print(wordMorph)
    if 'Mood=' not in wordMorph:
      continue
    mood = wordMorph.split('Mood=')[1]
    #print(mood)
    if '|' in mood:
      mood = mood.split('|')[0]
    #print(mood)
    assert mood != None
    moodCounts[mood] +=1
  #normalization
  numMoodForms = sum(moodCounts.values())
  if numMoodForms != 0:
    for key in moodCounts.keys():
      moodCounts[key] /= numMoodForms
  #assert list(verbFormCounts.keys()) == verbFormTypes
  for category in moodTypes:
    moodDescription = "Mood_Proportion:" + category
    linguisticFeatures[moodDescription] = moodCounts[category]
  return linguisticFeatures


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

def numFillers(df, fillers=['ah', 'eh', 'er', 'hm', 'mm', 'uh']):
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

"""SUBTLEX feats"""

subtlexCatDF = pd.read_csv(catalanLingFeatDirectory + "SUBTLEX-CAT.csv")
subtlexCatDF.head()
subtlexCatDF = subtlexCatDF.dropna()

wordSyllablesMapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['Num_Syl'])))
wordFrequencyMapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['Zipf'])))
wordContextualDiversityMapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['log(Rel_CD+1)'])))
wordColheartNMapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['N'])))
wordColheartNGreaterFreqMapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['NHF'])))
MBF_To_Mapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['MBF_To'])))
MBF_Ty_Mapping = dict(zip(list(subtlexCatDF['Words']), list(subtlexCatDF['MBF_Ty'])))

def subtlexMetrics(df,linguisticFeatures):
  dfNumLetters = []
  dfNumSyllables = []
  dfWordFreqs = []
  dfContextualDiversity = []
  dfColheartN = []
  dfColheartNGreaterFreq=[]
  dfLevenshtein = []
  dfMBF_To = []
  dfMBF_Ty = []
  for word in list(df['text']):
    wordNumSyllables = wordSyllablesMapping.get(word.lower())
    if wordNumSyllables != None:
      dfNumSyllables.append(wordNumSyllables)
    wordFreq = wordFrequencyMapping.get(word.lower())
    if wordFreq != None:
      dfWordFreqs.append(wordFreq)
    wordContextualDensity = wordContextualDiversityMapping.get(word.lower())
    if wordFreq != None:
      dfContextualDiversity.append(wordContextualDensity)
    wordColheartN = wordColheartNMapping.get(word.lower())
    if wordColheartN != None:
      dfColheartN.append(wordColheartN)
    wordCoheartGreaterFreq = wordColheartNGreaterFreqMapping.get(word.lower())
    if wordCoheartGreaterFreq!=None:
      wordCoheartGreaterFreq = wordCoheartGreaterFreq
      dfColheartNGreaterFreq.append(wordCoheartGreaterFreq)
    #wordLevenshtein = wordLevenshteinMapping(word.lower())
    #if wordLevenshtein !=None:
      #dfLevenshtein.append(wordLevenshtein)
    wordMBF_To = MBF_To_Mapping.get(word.lower())
    if wordMBF_To != None:
      if type(wordMBF_To) != float:
        wordMBF_To = wordMBF_To.replace(",","")
        wordMBF_To = float(wordMBF_To)
      dfMBF_To.append(wordMBF_To)
    wordMBF_Ty = MBF_Ty_Mapping.get(word.lower())
    if wordMBF_Ty !=None:
      if type(wordMBF_Ty) != float:
        wordMBF_Ty = wordMBF_Ty.replace(",", "")
        wordMBF_Ty = float(wordMBF_Ty             )
      dfMBF_Ty.append(wordMBF_Ty)
  linguisticFeatures['# of syllables mean'] = np.mean(dfNumSyllables)
  linguisticFeatures['Word Frequency mean'] = np.mean(dfWordFreqs)
  linguisticFeatures['Contextual Diversity mean'] = np.mean(dfContextualDiversity)
  linguisticFeatures["Coltheart's N mean"] = np.mean(dfColheartN)
  linguisticFeatures["NHF"] = np.mean(dfColheartNGreaterFreq)
  #df["Lev_N"] = np.mean(dfLevenshtein)
  #print(dfMBF_Ty)
  linguisticFeatures["MBF_To mean"] = np.nanmean(dfMBF_To)
  linguisticFeatures["MBF_Ty mean"] = np.nanmean(dfMBF_Ty)
  return linguisticFeatures


"""Imageability"""

import pandas as pd

imageabilityDF = pd.read_csv(catalanLingFeatDirectory +  "Catalan_Imageability_Translated_REVISED.txt",sep=' ', header=None, names=['Catalan', 'English', 'Imageability'],on_bad_lines='skip')
imageabilityDF[5:15] #observe how vaixwell has 3 imageability values, let's drop all of words with multiple imageability values
imageabilityDF = imageabilityDF[imageabilityDF['Catalan'] != imageabilityDF['English']]
imageabilityMapping = dict(zip(list(imageabilityDF['Catalan'].str.lower()), list(imageabilityDF['Imageability'])))

def imageability(df,linguisticFeatures):
  dfImageability = []
  for word in list(df['text']):
    wordImageability= imageabilityMapping.get(word.lower())
    if wordImageability != None:
      dfImageability.append(wordImageability)
  linguisticFeatures["Imageability mean"] = np.mean(dfImageability)
  #return np.mean(dfImageability)

"""Prevalence"""
prevalenceDF = pd.read_csv(catalanLingFeatDirectory + "Catalan_Imageability_Ratings.csv")
prevalenceMapping = dict(zip(list(prevalenceDF['Word']), list(prevalenceDF['Prevalence'])))
def prevalence(df,linguisticFeatures):
  dfPrevalence = []
  for word in list(df['text']):
    wordPrevalence = prevalenceMapping.get(word.lower())
    if wordPrevalence != None:
      dfPrevalence.append(wordPrevalence)
  linguisticFeatures["Prevalence mean"] = np.mean(dfPrevalence)
  #print(linguisticFeatures)
  #return np.mean(dfPrevalence)

"""Concreteness"""
concretenessDF = pd.read_csv(catalanLingFeatDirectory + "concreteness-estimates-ca.csv")
concretenessDF = concretenessDF[['word', 'estimated-concreteness']]
concretenessMapping = dict(zip(list(concretenessDF['word']), list(concretenessDF['estimated-concreteness'])))

def concreteness(df, linguisticFeatures):
  dfConcreteness = []
  for word in list(df['text']):
    wordConcreteness = concretenessMapping.get(word.lower())
    if wordConcreteness != None:
      dfConcreteness.append(wordConcreteness)
  linguisticFeatures["Concreteness mean"] = np.mean(dfConcreteness)
  #return linguisticFeatures

"""Word Length by Phonemes"""
import epitran
catalanEPI = epitran.Epitran('cat-Latn')

def wordLengthByPhonemes(df,linguisticFeatures):
  wordNumPhonemes = []
  for word in  list(df['text']):
    wordIPA = list(catalanEPI.transliterate(word))
    wordNumPhonemes.append(len(wordIPA))
  linguisticFeatures["Word length by phonemes mean"] = np.mean(wordNumPhonemes)
  #return np.mean(wordNumPhonemes)


"""Ratio of open class words to closed class words"""
def openClosedRatio(df,linguisticFeatures):
  open_class = {'VERB', 'NOUN', 'ADJ', 'ADV'}
  tag_types = list(df['POS'].unique())
  open_words = 0
  closed_words = 0
  for category in tag_types:
    if category in open_class:
      open_words += (df['POS'] == category).sum()
    else:
      closed_words += (df['POS'] == category).sum()
  if closed_words > 0:
    linguisticFeatures['Open Closed Words Ratio'] = open_words/closed_words
  else:
    linguisticFeatures['Open Closed Words Ratio'] = np.nan



"""# General Methods"""


def outputLinguisticFeatures(fileName):
  import csv
  csv_file = "Names.csv"
  with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = temp.keys())
    writer.writeheader()
    writer.writerow(temp)
    from google.colab import files
    files.download(csv_file)

def extractLinguisticFeatures(transcription):
  doc = nlp(transcription)#(filterText(transcription))
  df = setupSpacyDF(doc)
  linguisticFeatures = {}
  grammarComplexityIndex(df, linguisticFeatures)
  linguisticFeatures['Noun to Verb ratio'] = (df['POS'] == 'NOUN').sum()/(df['POS'] == 'VERB').sum()
  typeTokenRatio(df, linguisticFeatures)
  linguisticFeatures["Moving Average Type Token Ratio, n=5"] = mov_avg_type_token_ratio(df, window_size=5)
  linguisticFeatures["Moving Average Type Token Ratio, n=10"] = mov_avg_type_token_ratio(df, window_size=10)
  linguisticFeatures["Moving Average Type Token Ratio, n=15"] = mov_avg_type_token_ratio(df, window_size=15)
  linguisticFeatures["Moving Average Type Token Ratio, n=15"] = mov_avg_type_token_ratio(df, window_size=20)
  propositionalDensity(df, linguisticFeatures)
  posNormalizedCounts(df, linguisticFeatures)
  verbFormNormalizedCounts(df, linguisticFeatures)
  moodNormalizedCounts(df, linguisticFeatures)
  depNormalizedTagCounts(df, linguisticFeatures)
  averageSentenceLength(df, doc, linguisticFeatures)
  linguisticFeatures["# of Fillers per word"] = numFillers(df)/df.shape[0]
  linguisticFeatures["Repetitions per word"] = repetition_of_word_n(2,df)/df.shape[0]
  subtlexMetrics(df,linguisticFeatures)
  imageability(df,linguisticFeatures)
  prevalence(df, linguisticFeatures)
  concreteness(df,linguisticFeatures)
  #wordLengthByPhonemes(df,linguisticFeatures)
  #cohortEntropy(df,linguisticFeatures)
  openClosedRatio(df,linguisticFeatures)
  #phonotacticProbBigram(df,linguisticFeatures)
  #phonotacticProbUnigram(df,linguisticFeatures)
  return linguisticFeatures


def extractFeaturesFromFile(sampleName, transcription):
  lingFeats = extractLinguisticFeatures(transcription)
  lingFeatsDF = pd.Series(lingFeats).to_frame().T
  lingFeatsDF.index = [sampleName]
  lingFeatsDF.rename(columns = {'Unnamed: 0': 'file'}, inplace = True)
  return lingFeatsDF


def extractAWSIDForLinguistic(sampleName):
  return sampleName#.split('_')[0]

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
  directoryLinguisticFeats.insert(0,'Sample_ID', directoryLinguisticFeats['file'].apply(extractAWSIDForLinguistic))
  print("Shape of Linguistic Output:", directoryLinguisticFeats.shape)
  directoryLinguisticFeats.to_csv(fileName + "_LinguisticFeatures.csv", index=False)




runPipeline(inputDirectory, outputName)
