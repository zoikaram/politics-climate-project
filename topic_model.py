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

# Reading data 
files = glob.glob("./clean_data/*")
data = []
for i in range(len(files)):
    file = open(files[i])
    data.append([])
    for line in file.readlines():
        data[i].append(line.rstrip()) # rstrip removes the newline char
    file.close()

dictionary = Dictionary(data) 
corpus = [dictionary.doc2bow(text) for text in data]

lda = LdaMulticore(corpus, id2word=dictionary, num_topics=10, workers=11)
# dataframe = pd.DataFrame(data).transpose()
# for i in range(len(dataframe.columns)):
#     dataframe[i] = dataframe[i].value_counts().head(4000)

            
# topData = []
# threads = []
# for i in range(len(data)):
#     topData.append([])
#     thread = threading.Thread(target=keep_top_words, args=(data[i], topData[i], dataframe['vocab'].values)) 
#     threads.append(thread) # Store the thread object in a list so it does not get overwritten
#     thread.start() # start the thread object

# for thread in threads: # Once a thread finished it joins the workflow of the script again at this point
#     thread.join()