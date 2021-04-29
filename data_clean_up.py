# coding: utf-8 
# Data_clean_up.py
# This script processes the data into an output file which contains the cleaned up form of the data.

# Imports
import os
import glob
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import re
import threading
from datetime import datetime

# Defining methods for clean up of data
def alphanumeric(word):
    return re.sub(r'\W+', '', word)

def contains_non_ascii(s):
    return any(ord(i) > 127 for i in s)

def is_valid_word(word, stop_words=STOP_WORDS):
    """Checks if the word is valid for training or inference"""

    if stop_words and word in stop_words:
        return False

    if contains_non_ascii(word):
        return False

    distinct_keys = len(''.join(set(word)))
    if len(word) <= 2 or distinct_keys == 1:
        return False

    if word.isdigit() or word == "0":
        return False

    return True

# this function takes in an empty list which is fills with cleaned up data and a file for which it reads the data from
def document_cleaner(data, file):
    for word in nlp(file.read()):
        if(is_valid_word(word.text)):
            data.append(word.lemma_)
    
def folder_cleaner(data, files):
    for j in range(len(files)):
        file = open(files[j], "r", encoding="utf8")
        fileData = []
        document_cleaner(fileData, file)
        data.append(fileData)

print("start run:")
startTime = datetime.now().time()
print(startTime)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 1500000

print("Creating a list with all the names of the folders containing sessions")
folders = sorted(glob.glob("./UN-textdata/*")) # this list contains all the folder names in the data directory
print("Creating a matrix with all the names of the files with data")
files = [] # this list contains all the path names (relative to the working directory) of all the data files
for i in range(len(folders)):
    files.append(glob.glob(folders[i]+"/*.txt"))

print("Reading the data in the files to memory") 
threads = [] # this list holds all the threads used to compute the cleaned up data
data = [] # this list contains the cleaned up data
for i in range(len(folders)): # for each folder we
    #data.append([]) # 1: create a new list within the data list
    thread = threading.Thread(target=folder_cleaner, args=(data, files[i])) # 2: create a thread that reads the data of every file in that folder, cleans it up and stores it in the data list
    threads.append(thread) # 3: Store the thread object in a list so it does not get overwritten
    thread.start() # 4: start the thread object

for thread in threads: # Once a thread finished it joins the workflow of the script again at this point
    thread.join()
        


print("Cleanup complete")

outputFolder = "./clean_data"
try:
    for root, dirs, files in os.walk(outputFolder):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except:
                print(file+": no such file")
        os.rmdir(root)
    os.mkdir(outputFolder)
except FileExistsError:
    print("folder allready exists")
    
for i in range(len(data)):
    outputFile = open(outputFolder+"/clean_data_"+str(i), "w")
    for word in data[i]:
        outputFile.write(word)
        outputFile.write("\n")
    outputFile.close()

print("end run:")
endTime = startTime = datetime.now().time()
print(endTime)









