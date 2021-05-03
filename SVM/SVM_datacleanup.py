# coding: utf-8 
# Data_clean_up.py
# This script processes the data into an output file which contains the cleaned up form of the data.

# Imports
import glob

from spacy.lang.en.stop_words import STOP_WORDS
import re
from datetime import datetime
import nltk
import pandas as pd
from textblob import TextBlob
import concurrent.futures

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

print("start run:")
startTime = datetime.now().time()
print(startTime)

print("Creating a list with all the names of the folders containing sessions")
folders = sorted(glob.glob("./UN-textdata/*")) # this list contains all the folder names in the data directory
print("Creating a matrix with all the names of the files with data")
files = [] # this list contains all the path names (relative to the working directory) of all the data files
for i in range(len(folders)):
    files.append(glob.glob(folders[i]+"/*.txt"))

  
def folder_cleaner(folderName, files):
    data = pd.DataFrame(columns=['folder', 'file', 'sentence'])
    index = 0
    for i in range(len(files)):
        file = open(files[i], "r", encoding="utf8")
        sentences = nltk.sent_tokenize(file.read())
        for j in range(len(sentences)):
            index +=1
            data.loc[index, "folder"] = folderName
            data.loc[index, "file"] = file.name
            data.loc[index, "sentence"] = sentences[j]

    data['sentence'] = [entry.lower() for entry in data['sentence']]
    for index,entry in enumerate(data['sentence']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        for word, tag in TextBlob(entry).tags:
            if(is_valid_word(word) and tag=='NN'):
                Final_words.append(word.lemmatize())
        # The final processed set of words for each iteration will be stored in 'text_final'
        data.loc[index,'text_final'] = str(Final_words)
    print("folder cleaned")
    print(datetime.now().time())
    return data

data = pd.DataFrame(columns=["file", "sentence"]) # this list contains the cleaned up data
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    fs = []
    for i in range(len(folders)): 
        fs.append(executor.submit(folder_cleaner, folders[i], files[i]))
    concurrent.futures.wait(fs, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
    for future in fs:
        data = data.append(future.result(), ignore_index=True)
    
cleanedData = pd.DataFrame(columns=['folder', 'file', 'text'])
cleanedData['folder'] = data['folder']
cleanedData['file'] = data['file']
cleanedData['text'] = data['text_final']
cleanedData.to_csv("cleanUNData.csv")

print("end run:")
endTime = startTime = datetime.now().time()
print(endTime)


