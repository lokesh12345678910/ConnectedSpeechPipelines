# this tokeniser takes care of contractions nicely
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd

def filterText(text: str) -> str:
    """Remove unwanted characters and empty words

    Args:
        text (str): _description_

    Returns:
        str: The new string without the unwanted characters
    """
    # Create a reference variable for Class WhitespaceTokenizer
    tk = WhitespaceTokenizer()
    trackWords = tk.tokenize(text)
    #removing unwanted characters, excluding contractions
    bad_chars = [',', ':', '!', '\"', '?']
    filteredTrackWords = [''.join(filter(lambda i: i not in bad_chars, word)) for word in trackWords]
    #remove empty words
    filteredTrackWords = [i for i in filteredTrackWords if i] 
    
    return " ".join(filteredTrackWords)

def setupSpacyDF(doc) -> pd.DataFrame:
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

