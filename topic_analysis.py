# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:48:30 2021

@author: dries
"""

from gensim.test.utils import datapath
from gensim.models import LdaMulticore
import os
import glob
from gensim.corpora.dictionary import Dictionary

# Load pretrained model from disk.
saveFile = os.getcwd()+"/lda_files/lda_model"
ldaSave = datapath(saveFile)
lda = LdaMulticore.load(ldaSave)

# load example file
fileName = glob.glob("./UN-textdata/Session 73 - 2018/*.txt")
data = []
for i in range(len(fileName)):
    file = open(fileName[i], "r", encoding="utf8")
    data.append([])
    for line in file.readlines():
        for word in line.split():
            data[i].append(word)

dictionary = Dictionary(data) 
corpus = [dictionary.doc2bow(text) for text in data]
vector = lda[corpus[3]]