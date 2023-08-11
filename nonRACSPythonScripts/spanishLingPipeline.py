import sys

inputDirectory=sys.argv[1] # "/work/09424/smgrasso1/ls6/trialRuns/spanishPipelineTrialFiles/"
outputName=sys.argv[2] # "SpanishTrialPipelineOutput"
lingFeatDirectory ="/work/09424/smgrasso1/ls6/SpanishLingFeatData/"

import spacy_transformers
import spacy

nlp = spacy.load('es_dep_news_trf')
nlp = nlp.from_disk(lingFeatDirectory + "spanishSpaCyModel_July18")
import math
import pandas as pd
import numpy as np
import nltk
spanishLingFeatDirectory = lingFeatDirectory 
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
  #f = open(address,"r")
  #text = f.read()
  #doc = nlp(filterText(text))
  import pandas as pd
  cols = ("text",  "POS", "TAG", "STOP", "DEP", "MORPH")
  rows = []
  for t in doc:
    if t.pos_ == 'PUNCT' or t.pos_ == 'SYM' or t.pos_ == 'SPACE' or t.pos_ == 'X':
      #not considering punctuation
      continue
    row = [t.text, t.pos_, t.tag_, t.is_stop, t.dep_, t.morph]
    rows.append(row)
  df = pd.DataFrame(rows, columns=cols)
  return df


def posTagCounts(df, linguisticFeatures):
  #source of list of tags: scroll to bottom of https://stackoverflow.com/questions/58215855/how-to-get-full-list-of-pos-tag-and-dep-in-spacy
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
    tag_description = "POS_Proportion:" + category
    linguisticFeatures[tag_description] = (df['POS'] == category).sum()/df.shape[0]
  linguisticFeatures["POS_Proportion:" + "CONJ"] = linguisticFeatures["POS_Proportion:" + "CCONJ"] + linguisticFeatures["POS_Proportion:" + "SCONJ"]

def depTagCounts(df, linguisticFeatures):
  tag_types = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]
  for category in tag_types:
    depTagDescription = "DEP_Proportion:" + category
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
  linguisticFeatures["# of words"] = numTokens
  numTypes = len(set(df['text'].str.lower()))
  linguisticFeatures["# of unique words"] = numTypes
  typeTokenRatio = numTypes/numTokens
  linguisticFeatures["Type Token Ratio (# of unique words/# of words)"] = typeTokenRatio


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

textFileAddress = spanishLingFeatDirectory + "spanishCPdatabase2.txt"
cols  = 'Word Phono Length_Letters Length_Phonomes Frequency sOTAN sOTAF sOTAW sOTHN sOTHF sOTHW sOSAN sOSAF sOSAW sOSHN sOSHF sOSHW sODAN sODAF sODAW sODHN sODHF sODHW sOAAN sOAAF sOAAW sOAHN sOAHF sOAHW sPTAN sPTAF sPTAW sPTHN sPTHF sPTHW sPSAN sPSAF sPSAW sPSHN sPSHF sPSHW sPDAN sPDAF sPDAW sPDHN sPDHF sPDHW sPAAN sPAAF sPAAW sPAHN sPAHF sPAHW dOTAN dOTAF dOTAW dOTHN dOTHF dOTHW dOSAN dOSAF dOSAW dOSHN dOSHF dOSHW dODAN dODAF dODAW dODHN dODHF dODHW dOAAN dOAAF dOAAW dOAHN dOAHF dOAHW dPTAN dPTAF dPTAW dPTHN dPTHF dPTHW dPSAN dPSAF dPSAW dPSHN dPSHF dPSHW dPDAN dPDAF dPDAW dPDHN dPDHF dPDHW dPAAN dPAAF dPAAW dPAHN dPAHF dPAHW eOTAN eOTAF eOTAW eOTHN eOTHF eOTHW eOSAN eOSAF eOSAW eOSHN eOSHF eOSHW eODAN eODAF eODAW eODHN eODHF eODHW eOAAN eOAAF eOAAW eOAHN eOAHF eOAHW ePTAN ePTAF ePTAW ePTHN ePTHF ePTHW ePSAN ePSAF ePSAW ePSHN ePSHF ePSHW ePDAN ePDAF ePDAW ePDHN ePDHF ePDHW ePAAN ePAAF ePAAW ePAHN ePAHF ePAHW fOTAN fOTAF fOTAW fOTHN fOTHF fOTHW fOSAN fOSAF fOSAW fOSHN fOSHF fOSHW fODAN fODAF fODAW fODHN fODHF fODHW fOAAN fOAAF fOAAW fOAHN fOAHF fOAHW fPTAN fPTAF fPTAW fPTHN fPTHF fPTHW fPSAN fPSAF fPSAW fPSHN fPSHF fPSHW fPDAN fPDAF fPDAW fPDHN fPDHF fPDHW fPAAN fPAAF fPAAW fPAHN fPAHF fPAHW gOTAN gOTAF gOTAW gOTHN gOTHF gOTHW gOSAN gOSAF gOSAW gOSHN gOSHF gOSHW gODAN gODAF gODAW gODHN gODHF gODHW gOAAN gOAAF gOAAW gOAHN gOAHF gOAHW gPTAN gPTAF gPTAW gPTHN gPTHF gPTHW gPSAN gPSAF gPSAW gPSHN gPSHF gPSHW gPDAN gPDAF gPDAW gPDHN gPDHF gPDHW gPAAN gPAAF gPAAW gPAHN gPAHF gPAHW'
cols =cols.split()
clearPondDF = pd.read_csv(textFileAddress,delim_whitespace=True, encoding='latin-1', on_bad_lines='skip',names=cols)
import math
clearPondDF['Log_Frequency'] = clearPondDF['Frequency'].apply(math.log)
wordNumPhonemeMapping = dict(zip(list(clearPondDF['Word']), list(clearPondDF['Length_Phonomes'])))
wordFrequencyMapping = dict(zip(list(clearPondDF['Word']), list(clearPondDF['Log_Frequency'])))

##Set up for Cohort Entropy

###**Computing Initial Phoneme Distribution**
def splitPhonemes(x):
  return x.split('.'  )
clearPondDF['Phono'] = clearPondDF['Phono'].apply(splitPhonemes)
initialPhonemeCount={}
for decomp in list(clearPondDF['Phono']):
  firstPhoneme = decomp[0]
  numOccurences = initialPhonemeCount.get(firstPhoneme)
  if numOccurences != None:
    initialPhonemeCount[firstPhoneme] = numOccurences + 1
  else:
    initialPhonemeCount[firstPhoneme] = 1
dictLength = sum(initialPhonemeCount.values(), 0.0)
initialPhonemeDistribution = {k: v / dictLength for k, v in initialPhonemeCount.items()}
smallestValForFirstPhoneme = min(initialPhonemeDistribution.values())


###**1st Transition Matrix: P(Phoneme 2|Phoneme 1)**

allPhonemes = [decomp for decomp in list(clearPondDF['Phono'])]
allPhonemes = set()
for decomp in list(clearPondDF['Phono']):
  allPhonemes.update(decomp)
phonemeMapping = {}
allPhonemes = list(allPhonemes)
for i in range(len(allPhonemes)):
  phoneme = allPhonemes[i]
  phonemeMapping[phoneme] = i
numPhonemes = len(allPhonemes)
import numpy as np
firstTransitionCountMatrix = np.zeros([numPhonemes, numPhonemes])
for wordDecomp in list(clearPondDF['Phono']):
  wordPhonemes = wordDecomp
  firstTransitionPair = wordPhonemes[0:2]
  #special case: word only has one phoneme
  if len(firstTransitionPair) == 1:
    continue
  #print(firstTransitionPair)
  firstPhonemeIndex = phonemeMapping.get(firstTransitionPair[0])
  secondPhonemeIndex = phonemeMapping.get(firstTransitionPair[1])
  assert firstPhonemeIndex != None, print(firstTransitionPair[0], "is not in phoneme mapping dict")
  assert secondPhonemeIndex != None, print(firstTransitionPair[1], "is not in phoneme mapping dict")
  firstTransitionCountMatrix[firstPhonemeIndex][secondPhonemeIndex] +=1


"""normalize each row"""
from sklearn.preprocessing import normalize
firstTransitionProbMatrix = normalize(firstTransitionCountMatrix, axis=1, norm='l1')

###**Main Transition Matrix: P(Phoneme n | phoneme n-1, phoneme n-2)**
mainTransitionCountMatrix = np.zeros([numPhonemes*numPhonemes, numPhonemes])
allPhonemePairs= list(((x, y) for x in allPhonemes for y in allPhonemes))
phonemePairMapping = {}
for i in range(len(allPhonemePairs)):
  pair = allPhonemePairs[i]
  phonemePairMapping[pair] = i
clearPondDF['Phono'].apply(len)
for wordDecomp in list(clearPondDF['Phono']):
  list_of_triplet = [wordPhonemes[i:i+3] for i in range(0, len(wordPhonemes) - 2)]
  for triplet in list_of_triplet:
    #print(triplet[0:2])
    priorPhonemePairIndex = phonemePairMapping.get(tuple(triplet[0:2]))
    nextPhonemeIndex = phonemeMapping.get(triplet[-1])
    assert priorPhonemePairIndex != None, print(triplet[0:2], "is not in phoneme pair mapping dict")
    assert secondPhonemeIndex != None, print(triplet[-1], "is not in phoneme mapping dict")
    mainTransitionCountMatrix[priorPhonemePairIndex][nextPhonemeIndex] +=1

from sklearn.preprocessing import normalize
mainTransitionProbMatrix = normalize(mainTransitionCountMatrix, axis=1, norm='l1')
smallestValForMainTransition = min(mainTransitionProbMatrix[np.nonzero(mainTransitionProbMatrix)])
smallestValForInitialTransition = min(firstTransitionProbMatrix[np.nonzero(firstTransitionProbMatrix)])


def calculateAtPhonemeLevel(trackPhonemes):
  trackPhonemes = [phoneme.strip() for phoneme in trackPhonemes]
  firstPhoneme = trackPhonemes[0]
  firstPhonemeSuprisal = None
  firstPhonemeSuprisal = initialPhonemeDistribution.get(firstPhoneme)
  if firstPhonemeSuprisal == 0 or firstPhonemeSuprisal == None:
    firstPhonemeSuprisal = math.log(smallestValForFirstPhoneme **2)
    return [firstPhonemeSuprisal]
  else:
    firstPhonemeSuprisal = math.log(firstPhonemeSuprisal)
    return [firstPhonemeSuprisal]
  secondPhoneme = trackPhonemes[1]
  secondPhonemeSuprisal = None
  secondPhonemeSuprisal = computeFirstTransition(firstPhoneme, secondPhoneme)
  trackPhonemeSuprisalVals = [firstPhonemeSuprisal, secondPhonemeSuprisal]
  #indexedTrackPhonemes = [phonemeMapping[phoneme.upper()] for phoneme in trackPhonemes]
  for i in range(2, len(trackPhonemes)):
    currentPhoneme = trackPhonemes[i].upper()
    priorTwoPhonemes = [phoneme.upper() for phoneme in trackPhonemes[i-2:i]]
    currentPhonemeSurprisal = None
    currentPhonemeSuprisal = computeSurprisalForPhoneme(currentPhoneme, priorTwoPhonemes)
    trackPhonemeSuprisalVals.append(currentPhonemeSuprisal)
  return trackPhonemeSuprisalVals

def computeSurprisalForPhoneme(currentPhoneme, priorTwoPhonemesPair):
  currentPhonemeIndex = phonemeMapping.get(currentPhoneme)
  priorPhonemePairIndex = phonemePairMapping.get(tuple(priorTwoPhonemesPair))
  firstPhonemeOfPair = priorTwoPhonemesPair[0]
  secondPhonemeOfPair = priorTwoPhonemesPair[1]
  currentPhonemeSuprisal = None
  if currentPhonemeIndex == None or priorPhonemePairIndex == None:
    return math.log(smallestValForInitialTransition **2)
  currentPhonemeSuprisal = mainTransitionProbMatrix[priorPhonemePairIndex][currentPhonemeIndex]
  print(currentPhonemeSuprisal)
  #print(currentPhoneme, priorTwoPhonemesPair,currentPhonemeSuprisal)
  #assert type(currentPhonemeSuprisal) == float
  if currentPhonemeSuprisal == 0:
      currentPhonemeSuprisal= smallestValForMainTransition **2
  return math.log(currentPhonemeSuprisal)

def computeFirstTransition(firstPhoneme, secondPhoneme):
  firstPhonemeIndex = phonemeMapping[firstPhoneme]
  secondPhonemeIndex = phonemeMapping[secondPhoneme]
  secondPhonemeSuprisal = firstTransitionProbMatrix[firstPhonemeIndex][secondPhonemeIndex]
  if secondPhonemeSuprisal == 0 or secondPhonemeSuprisal == None:
    secondPhonemeSuprisal = math.log(smallestValForInitialTransition **2)
  else:
    secondPhonemeSuprisal = math.log(secondPhonemeSuprisal)
  return secondPhonemeSuprisal

wordPhonemeMapping = dict(zip(list(clearPondDF['Word']), list(clearPondDF['Phono'])))

def phonemicSurprisal(df,linguisticFeatures):
  dfPhonemes = []
  phonemeLevelSurprisalValues = []
  for word in list(df['text']):
    wordPhonemes = wordPhonemeMapping.get(word)
    if wordPhonemes != None:
      for phoneme in wordPhonemes:
        dfPhonemes.append(phoneme)
      phonemeLevelSurprisalValues.append(calculateAtPhonemeLevel(wordPhonemes))
  #print(dfPhonemes)
  dfPhonemicSuprisalVals = calculateAtPhonemeLevel(dfPhonemes)
  linguisticFeatures["Phonemic Surprisal Word Level Mean"] = np.mean(dfPhonemicSuprisalVals)
  linguisticFeatures["Phonemic Surprisal Phoneme Level Mean"] = np.mean(phonemeLevelSurprisalValues)


"""Cohort Entropy"""
wordPhonemeMapping = dict(zip(list(clearPondDF['Word']), list(clearPondDF['Phono'])))
def getAllWordsWithPriorPhonemes(phonemeDecompToMatch):
  matchingWords = []
  for word in wordPhonemeMapping.keys():
    wordPhonemeDecomp = wordPhonemeMapping[word]
    if len(wordPhonemeDecomp) < len(phonemeDecompToMatch):
      continue
    if wordPhonemeDecomp[0:len(phonemeDecompToMatch)] == phonemeDecompToMatch:
      matchingWords.append(word)
  return matchingWords
clearPondDF['WordProb'] = clearPondDF['Frequency']/sum(clearPondDF['Frequency'])
clearPondDF['LogWordProb'] = clearPondDF['WordProb'].apply(math.log)
wordProbs = dict(zip(clearPondDF['Word'], clearPondDF['WordProb']))
logWordProbs = dict(zip(clearPondDF['Word'], clearPondDF['LogWordProb']))
smoothProb = 1/(clearPondDF.shape[0]+1)
smoothLogProb = math.log(smoothProb)
smoothProb * smoothLogProb


def cohortEntropyForWordPhonemes(wordPhonemicSequence):
  wordCohortEntropyVals = []
  pastPhonemes = [wordPhonemicSequence[0]]
  for phoneme in wordPhonemicSequence[1:]:
    cohort = getAllWordsWithPriorPhonemes(pastPhonemes)
    #print("# of words starting with", pastPhonemes, ":", len(cohort))
    cohortEntropy = 0

    for word in cohort:
      uppercaseWord = word[0].upper() + word[1:]
      lowercaseWord = word.lower()
      logWordProb = np.nanmax(np.array([smoothLogProb, logWordProbs.get(lowercaseWord), logWordProbs.get(uppercaseWord)], dtype=np.float64))
      wordProb = np.nanmax(np.array([smoothProb, wordProbs.get(lowercaseWord), wordProbs.get(uppercaseWord)], dtype=np.float64))
      cohortEntropy += wordProb * logWordProb
    if len(cohort) == 0:
       cohortEntropy = smoothProb * smoothLogProb
    wordCohortEntropyVals.append(cohortEntropy)
    pastPhonemes.append(phoneme)
  return wordCohortEntropyVals


def cohortEntropy(df,linguisticFeatures):
  dfCohortEntropyValues = []
  phonemeLevelCohortEntropyValues = []
  for word in list(df['text']):
    wordPhonemes = wordPhonemeMapping.get(word)
    if wordPhonemes == None:
      continue
    phonemesInWordCohortEntropyVals = cohortEntropyForWordPhonemes(wordPhonemes)
    for val in phonemesInWordCohortEntropyVals:
      phonemeLevelCohortEntropyValues.append(val)
    if len(phonemesInWordCohortEntropyVals) != 0:
      wordLevelCohortEntropy = np.mean(phonemesInWordCohortEntropyVals)
      dfCohortEntropyValues.append(wordLevelCohortEntropy)
  linguisticFeatures["Cohort Entropy Word Level Mean"] = np.mean(dfCohortEntropyValues)
  linguisticFeatures["Cohort Entropy Phoneme Level Mean"] = np.mean(phonemeLevelCohortEntropyValues)
  #return linguisticFeatures
  #linguisticFeatures["Cohort Entropy SD"] = np.std(dfCohortEntropyValues)

"""Phonotactic Probability"""

ipa_to_Klattese_map = {    "?": "C",
    "?": "J",
    "?": "S",
    "?": "Z",
    "?": "T",
    "?": "G",
    "j": "y",
    "?": "I",
    "?": "E",
    "?": "a",
    "a?": "W",
    "a?": "Y",
    "?": "^",
    "?": "c",
    "o?": "O",
    "?": "U",
    "?r": "R",
    "?": "x",
    "'": ""}
import requests
from bs4 import BeautifulSoup
spanishEPI = epitran.Epitran('spa-Latn')

def calculatePhonotaticProbability(word):
  word_ipa = spanishEPI.transliterate(word)
  #handle two characters ipa
  word_ipa = list(word_ipa)
  for i in range(len(word_ipa)-1):
    if (word_ipa[i]+word_ipa[i+1]) in ipa_to_Klattese_map.keys():
      word_ipa[i] = ipa_to_Klattese_map[word_ipa[i]+word_ipa[i+1]]
      word_ipa[i+1] = ""

  for i in range(len(word_ipa)):
    if word_ipa[i] in ipa_to_Klattese_map.keys():
      word_ipa[i] = ipa_to_Klattese_map[word_ipa[i]]
  text = "".join(word_ipa)
  url = "https://calculator.ku.edu/phonotactic/Spanish/output?words="+text
  res=requests.get(url)
  soup = BeautifulSoup(res.text, "html.parser")
  table = soup.findAll('strong')
  #print(word_ipa)
  if(len(table)==0):
    return None,None
    print("the ipa is "+ word_ipa)
    print("after replace the Klattese is " + "".join(word_ipa))
    print(word + "is not converted to Klattese well, please fix ipa-Klattese dict better")
  one_gram_result = table[0].get_text().split()[0]
  two_gram_result = table[0].get_text().split()[1]
  return one_gram_result, two_gram_result

def phonotacticProbability(df,linguisticFeatures):
  wordLevelPhonotacticProbOneGram = []
  wordLevelPhonotacticProbTwoGram = []
  for word in list(df['text']):
    one_gram_result, two_gram_result = calculatePhonotaticProbability(word)
    #print(one_gram_result,two_gram_result)
    if one_gram_result != None:
      try:
        one_gram_result = float(one_gram_result)
        wordLevelPhonotacticProbOneGram.append(one_gram_result)
      except:
        one_gram_result = None #added random line to avoid compile error
    if two_gram_result != None:# and type(two_gram_result) !=str :
      try:
        two_gram_result = float(two_gram_result)
        wordLevelPhonotacticProbTwoGram.append(two_gram_result)
      except:
        one_gram_result = None #added random line to avoid compile error
  assert len(wordLevelPhonotacticProbOneGram) !=0
  assert len(wordLevelPhonotacticProbTwoGram) !=0
  linguisticFeatures["Phonotactic Probability Unigram Mean"] = np.mean(wordLevelPhonotacticProbOneGram)
  linguisticFeatures["Phonotactic Probability Bigram Mean"] = np.mean(wordLevelPhonotacticProbTwoGram)

"""Affective Ratings:Valence, Arousal"""
affectivenessDF = pd.read_csv(spanishLingFeatDirectory + 'BR-Org-15-242R1 Stadthagen et al Spanish Emotional Norms BRM.csv', encoding='latin-1')

def computeAffectiveRatingsForWordList(df,linguisticFeatures):
  affectiveRatingsForEachWord = []
  for word in list(df['text']):
    wordAffectiveRatingsDF = affectivenessDF[affectivenessDF['Word'] == word].iloc[:,1:]
    if wordAffectiveRatingsDF.shape[0] == 0:
      continue
    wordAffectiveRatings = np.array(wordAffectiveRatingsDF)[0]
    affectiveRatingsForEachWord.append(wordAffectiveRatings)
  affectiveRatingsForEachWord = np.array(affectiveRatingsForEachWord)
  affectiveRatingsAveragedAcrossEachWord = np.mean(affectiveRatingsForEachWord,axis=0)
  #return affectiveRatingsForEachWord
  for affectiveRating, average in zip(affectivenessDF.columns[1:],affectiveRatingsAveragedAcrossEachWord):
    linguisticFeatures['Word ' + affectiveRating + ' Average'] = average

"""Ratio of Open Class Words to Closed Class Words"""
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


"""ESPAL feats"""

import pandas as pd
espalDF = pd.read_csv(spanishLingFeatDirectory + 'espal.csv')
espalDF = espalDF.fillna("\\N")

wordFamilarityMapping = dict(zip(list(espalDF['word']), list(espalDF['familiarity'])))
wordImageabilityMapping = dict(zip(list(espalDF['word']), list(espalDF['imageability'])))
wordConcretenessMapping = dict(zip(list(espalDF['word']), list(espalDF['concreteness'])))
wordNumSyllablesMapping = dict(zip(list(espalDF['word']), list(espalDF['es_num_syll'])))
wordNumPhonemesMapping = dict(zip(list(espalDF['word']), list(espalDF['es_num_phon'])))
wordFreqMapping = dict(zip(list(espalDF['word']), list(espalDF['frq'])))
wordLevenshteinMapping = dict(zip(list(espalDF['word']), list(espalDF['Lev_N'])))
wordNHFMapping = dict(zip(list(espalDF['word']), list(espalDF['NHF'])))
pos_MBF_To_Mapping = dict(zip(list(espalDF['word']), list(espalDF['pos_tok_MBOF'])))
abs_MBF_To_Mapping = dict(zip(list(espalDF['word']), list(espalDF['abs_tok_MBOF'])))
pos_MBF_Ty_Mapping = dict(zip(list(espalDF['word']), list(espalDF['pos_type_MBOF'])))
abs_MBF_Ty_Mapping = dict(zip(list(espalDF['word']), list(espalDF['abs_type_MBOF'])))

def espalMetrics(df,linguisticFeatures):
  dfNumPhonemes = []
  dfWordFreqs = []
  dfWordFamiliarity = []
  dfWordImageability = []
  dfWordConcreteness = []
  dfWordNumSyllables = []
  dfNHF=[]
  dfLevenshtein = []
  df_pos_MBF_To = []
  df_abs_MBF_To = []
  df_pos_MBF_Ty = []
  df_abs_MBF_Ty = []
  for word in list(df['text']):
    wordNHF = wordNHFMapping.get(word.lower())
    if wordNHF != None and wordNHF != '\\N':
      dfNHF.append(wordNHF)
    wordLevenshtein = wordLevenshteinMapping.get(word.lower())
    if wordLevenshtein != None and wordLevenshtein != '\\N':
      dfLevenshtein.append(wordLevenshtein)
    word_pos_MBF_To = pos_MBF_To_Mapping.get(word.lower())
    if word_pos_MBF_To != None and word_pos_MBF_To != '\\N':
      df_pos_MBF_To.append(word_pos_MBF_To)
    word_abs_MBF_To = abs_MBF_To_Mapping.get(word.lower())
    if word_abs_MBF_To != None and word_abs_MBF_To != '\\N':
      df_abs_MBF_To.append(word_abs_MBF_To)
    word_pos_MBF_Ty = pos_MBF_To_Mapping.get(word.lower())
    if word_pos_MBF_Ty != None and word_pos_MBF_Ty != '\\N':
      df_pos_MBF_Ty.append(word_pos_MBF_Ty)
    word_abs_MBF_Ty = abs_MBF_Ty_Mapping.get(word.lower())
    if word_abs_MBF_Ty != None and word_abs_MBF_Ty != '\\N':
      df_abs_MBF_Ty.append(word_abs_MBF_Ty)
    wordNumPhonemes = wordNumPhonemesMapping.get(word.lower())
    if wordNumPhonemes != None and wordNumPhonemes != '\\N':
      dfNumPhonemes.append(wordNumPhonemes)
    wordFreq = wordFreqMapping.get(word.lower())
    if wordFreq != None and wordFreq != '\\N':
      dfWordFreqs.append(float(wordFreq))
    wordFamiliarity = wordFamilarityMapping.get(word.lower())
    if wordFamiliarity != None and wordFamiliarity != '\\N':
      dfWordFamiliarity.append(float(wordFamiliarity))
    wordImageability = wordImageabilityMapping.get(word.lower())
    if wordImageability != None and wordImageability != '\\N':
      dfWordImageability.append(float(wordImageability))
    wordConcreteness = wordConcretenessMapping.get(word.lower())
    if wordConcreteness != None and wordConcreteness != '\\N':
      dfWordConcreteness.append(float(wordConcreteness))
    wordNumSyllables = wordNumSyllablesMapping.get(word.lower())
    if wordNumSyllables != None and wordNumSyllables != '\\N':
      dfWordNumSyllables.append(wordNumSyllables)
  linguisticFeatures["Word length by Phonemes"] = np.mean(dfNumPhonemes)
  linguisticFeatures["Word Frequency per million"] = np.mean(dfWordFreqs)
  linguisticFeatures["Word Familiarity"] = np.mean(dfWordFamiliarity)
  linguisticFeatures["Word Imageability"] = np.mean(dfWordImageability)
  linguisticFeatures["Word Concreteness"] = np.mean(dfWordConcreteness)
  linguisticFeatures["Word length by syllables"] = np.mean(dfWordNumSyllables)
  linguisticFeatures["NHF"] = np.mean(dfNHF)
  linguisticFeatures["Lev_N"] = np.mean(dfLevenshtein)
  linguisticFeatures["MBF_To_pos"] = np.mean(df_pos_MBF_To)
  linguisticFeatures["MBF_To_abs"] = np.mean(df_abs_MBF_To)
  linguisticFeatures["MBF_Ty_pos"] = np.mean(df_pos_MBF_Ty)
  linguisticFeatures["MBF_Ty_abs"] = np.mean(df_abs_MBF_Ty)
  return linguisticFeatures
  print("# of phonemes for each word:",dfNumPhonemes)
  print(np.min(dfNumPhonemes),np.max(dfNumPhonemes),np.mean(dfNumPhonemes),np.std(dfNumPhonemes))
  print(np.min(dfWordFreqs),np.max(dfWordFreqs),np.mean(dfWordFreqs),np.std(dfWordFreqs))
  #return np.min(wordNumPhonemes),np.max(wordNumPhonemes),np.mean(wordNumPhonemes),np.std(wordNumPhonemes)


"""# Prevalence"""

import pandas as pd
import os
prevalenceDF = pd.read_csv(spanishLingFeatDirectory + 'spanishPrevalenceData.csv')
prevalenceMapping = dict(zip(list(prevalenceDF['spelling']), list(prevalenceDF['prevalence_total'])))

def prevalence(df,linguisticFeatures):
  df_prevalence = []
  for word in list(df['text']):
    wordPrevalence = prevalenceMapping.get(word.lower())
    if wordPrevalence !=None:
      #print(word, wordPrevalence)
      df_prevalence.append(wordPrevalence)
  linguisticFeatures["Prevalence mean"] = np.mean(df_prevalence)

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
  df = setupSpacyDF(doc )
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
  #clearPondMetrics(df,linguisticFeatures) #covered by ESPAL
  #wordLengthByMorphemes(df,linguisticFeatures)
  computeAffectiveRatingsForWordList(df,linguisticFeatures)
  phonemicSurprisal(df,linguisticFeatures)
  #cohortEntropy(df,linguisticFeatures)
  #phonotacticProbability(df,linguisticFeatures)
  openClosedRatio(df,linguisticFeatures)
  espalMetrics(df,linguisticFeatures)
  prevalence(df,linguisticFeatures)
  return linguisticFeatures

def extractFeaturesFromFile(sampleName, transcription):
  lingFeats = extractLinguisticFeatures(transcription)
  lingFeatsDF = pd.Series(lingFeats).to_frame().T
  lingFeatsDF.index = [sampleName]
  lingFeatsDF.rename(columns = {'Unnamed: 0': 'file'}, inplace = True)
  return lingFeatsDF


def extractAWSIDForLinguistic(sampleName):
  return sampleName
  #return sampleName.split('_')[0]

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
