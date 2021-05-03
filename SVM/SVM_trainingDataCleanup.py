# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:49:47 2021

@author: dries
"""

# importing required libraries
import pandas as pd
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS
import re


# Defining methods for clean up of data
def alphanumeric(word):
    return re.sub(r'\W+', '', word)

def contains_non_ascii(s):
    return any(ord(i) > 127 for i in s)

def is_valid_word(word, stop_words=STOP_WORDS):
    """Checks if the word is valid for training or inference"""

    if word in stop_words:
        return False

    if contains_non_ascii(word):
        return False

    distinct_keys = len(''.join(set(word)))
    if len(word) <= 2 or distinct_keys == 1:
        return False

    if word.isdigit() or word == "0":
        return False

    return True

# reading tsv file
corpus = pd.read_table("./TrainingData.tsv", sep="\t")

corpus['sentence'] = [entry.lower() for entry in corpus['sentence']]
for index,entry in enumerate(corpus['sentence']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    for word, tag in TextBlob(entry).tags:
        if(is_valid_word(word) and tag=='NN'):
            Final_words.append(word.lemmatize())
    # The final processed set of words for each iteration will be stored in 'text_final'
    corpus.loc[index,'text_final'] = str(Final_words)

cleanedData = pd.DataFrame(columns=['label', 'text'])
cleanedData['label'] = corpus['label']
cleanedData['text'] = corpus['text_final']
cleanedData.to_csv("cleanTrainingData.csv")











