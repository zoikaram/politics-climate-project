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
usedData = []

dictionary = Dictionary(data) 
corpus = [dictionary.doc2bow(text) for text in data]
if __name__ == '__main__':
    lda = LdaMulticore(corpus, id2word=dictionary, num_topics=15, 
                                        chunksize=500,
                                        update_every=1,
                                        passes=500,
                                        iterations=5000,
                                        alpha = 'asymmetric',
                                        eta=0.0001,
                                        random_state = 100
                                        )
    # best model so far
    # lda = LdaMulticore(corpus, id2word=dictionary, num_topics=15, 
    #                                     chunksize=5,
    #                                     passes=500,
    #                                     iterations=5000,
    #                                     alpha = 'asymmetric',
                                        
    #                                     )

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
