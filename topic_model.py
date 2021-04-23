# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:22:41 2021

@author: dries
"""
# This script performs an LDA analysis on a clean set of data
# importing modules
import glob
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.test.utils import datapath
from datetime import datetime
import os
from collections import Counter

print("start run:")
startTime = datetime.now().time()
print(startTime)
# Reading data 
files = glob.glob("./clean_data/*")
data = []
for i in range(len(files)):
    file = open(files[i])
    data.append([])
    for line in file.readlines():
        data[i].append(line.rstrip()) # rstrip removes the newline char
    file.close()
usedData = data[1:-1]
#### tryout
concatdata = []
for doc in usedData:
    for word in doc:
        concatdata.append(word)
topWords = []
for word in Counter(concatdata).most_common(5000):
    topWords.append(word[0])
truncatedUsedData = []
for i in range(len(usedData)):
    truncatedUsedData.append([])
    for word in usedData[i]:
        if word in topWords:
            truncatedUsedData[i].append(word)
#### end tryout

dictionary = Dictionary(truncatedUsedData) 
corpus = [dictionary.doc2bow(text) for text in truncatedUsedData]
if __name__ == '__main__':
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=20, 
                                        chunksize=100,
                                        passes=10,
                                        iterations=150,
                                        alpha = 0.01)

    # saving model to disk
    saveFolder = os.getcwd()+"/lda_files"
    try:
        for root, dirs, files in os.walk(saveFolder):
            for file in files:
                try:
                    os.remove(os.path.join(root, file))
                except:
                    print(file+": no such file")
            os.rmdir(root)
        os.mkdir(saveFolder)
    except:
        print("save folder creation failed")    
    ldaSave = datapath(saveFolder+"/lda_model")
    lda.save(ldaSave)

    print("end run:")
    endTime = startTime = datetime.now().time()
    print(endTime)
