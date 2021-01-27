# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:26:25 2019

@author: HP
"""

import en_core_web_sm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import isfile, join
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from datetime import datetime
startTime = datetime.now()
print(startTime)
#path=input("Please enter the path of '20news-bydate-train' directory. For example: D:/20news-bydate.tar/20news-bydate-train")
nlp=en_core_web_sm.load()
segmentedSentenceAutos=[]
segmentedSentenceHockey=[]
pathTrain="D:/20news-bydate.tar/20news-bydate-train"
pathTest="D:/20news-bydate.tar/20news-bydate-test"
def preprocessingData(setname,path,documents):
    recAutosPath=path+"/rec.autos"
    recHockeyPath=path+"/rec.sport.hockey"
    segmentedSentenceAutosList=[]
    tokenizedWordsAutos=[]
    tokenizedWordsAutosList=[]
    segmentedSentenceHockeyList=[]
    tokenizedWordsHockey=[]
    tokenizedWordsHockeyList=[]
    uniqueWordTypes=[]
    uniqueWords=[]
    allDocs=[]
    filesAutos=[files for files in listdir(recAutosPath) if isfile(join(path+"/rec.autos", files))]
    filesHockey=[files for files in listdir(recHockeyPath) if isfile(join(path+"/rec.sport.hockey", files))]
    for eachFile in filesAutos:
        document=open(recAutosPath+"/"+eachFile).read()
        doc=nlp(document)
        for sent in doc.sents:
            segmentedSentenceAutosList.append(sent.text)            
        for token in doc:
            if token.is_stop is True:
                continue
            else:    
                tokenizedWordsAutosList.append((token.lemma_).lower())
                allDocs.append((token.lemma_).lower())
                uniqueWordTypes.append(token.pos_)
        segmentedSentenceAutos.append(segmentedSentenceAutosList)
        tokenizedWordsAutos.append(tokenizedWordsAutosList)

    for eachFile in filesHockey:
        document=open(recHockeyPath+"/"+eachFile).read()
        doc=nlp(document)
        for sent in doc.sents:
            segmentedSentenceHockeyList.append(sent.text)
        for token in doc:
            if token.is_stop is True:
                continue
            else:    
                tokenizedWordsHockeyList.append((token.lemma_).lower())
                allDocs.append((token.lemma_).lower())
                uniqueWordTypes.append(token.pos_)
        segmentedSentenceHockey.append(segmentedSentenceHockeyList)
        tokenizedWordsHockey.append(tokenizedWordsHockeyList)
    
    for i in tokenizedWordsAutos:
        documents.append(i)
    for i in tokenizedWordsHockey:
        documents.append(i)
    uniqueWordTypes=list(np.unique(uniqueWordTypes))
    uniqueWords=list(set(token for l in documents for token in l))
    print("Number of documents in rec.autos in ",setname,":",len(tokenizedWordsAutos))
    print("Number of documents in rec.sports.hockey in ",setname,":",len(tokenizedWordsHockey))
    print("Number of documents in the",setname,":",len(documents))
    print("Number of unique word types in the",setname,":",len(uniqueWordTypes))
    print("Number of unique words in the",setname,":",len(uniqueWords))
    return documents

def trainingData():
    documents=[]
    documents=preprocessingData("training set",pathTrain,documents)

def testingData():
    documents=[]
    documents=preprocessingData("testing set",pathTest,documents)

def preprocessTrainData():
    lists = []
    unique_words = []
    unique_word_types = []
    string = ''
    recAutosPath=pathTrain+"/rec.autos"
    filesAutos=[files for files in listdir(recAutosPath) if isfile(join(pathTrain+"/rec.autos", files))]
    for eachFile in filesAutos:
        document=open(recAutosPath+"/"+eachFile).read()
        myfile = nlp(document)
        for token in (myfile):
            if not token.is_stop:
                string =string+" "+(token.lemma_).lower()
                if (token.lemma_).lower() not in unique_words:
                    unique_words.append((token.lemma_).lower())
                tk=token.pos_
                if tk not in unique_word_types:
                    unique_word_types.append(tk)
        lists.append(string)
    recHockeyPath=pathTrain+"/rec.sport.hockey"
    filesHockey=[files for files in listdir(recHockeyPath) if isfile(join(pathTrain+"/rec.sport.hockey", files))]
    for eachFile in filesHockey:
        document=open(recHockeyPath+"/"+eachFile).read()
        myfile = nlp(document)
        for token in (myfile):
            if not token.is_stop:
                string =string+" "+(token.lemma_).lower()
                if (token.lemma_).lower() not in unique_words:
                    unique_words.append((token.lemma_).lower())
                tk=token.pos_
                if tk not in unique_word_types:
                    unique_word_types.append(tk)
        lists.append(string)
    return lists                    
                        
def preprocessTestData():
    lists = []
    unique_words = []
    unique_word_types = []
    string = ''
    recAutosPath=pathTest+"/rec.autos"
    filesAutos=[files for files in listdir(recAutosPath) if isfile(join(pathTest+"/rec.autos", files))]
    for eachFile in filesAutos:
        document=open(recAutosPath+"/"+eachFile).read()
        myfile = nlp(document)
        for token in (myfile):
            if not token.is_stop:
                string =string+" "+(token.lemma_).lower()
                if (token.lemma_).lower() not in unique_words:
                    unique_words.append((token.lemma_).lower())
                tk= token.pos_
                if tk not in unique_word_types:
                    unique_word_types.append(tk)
        lists.append(string)
    recHockeyPath=pathTest+"/rec.sport.hockey"
    filesHockey=[files for files in listdir(recHockeyPath) if isfile(join(pathTest+"/rec.sport.hockey", files))]
    for eachFile in filesHockey:
        document=open(recHockeyPath+"/"+eachFile).read()
        myfile = nlp(document)
        for token in (myfile):
            if not token.is_stop:
                string =string+" "+(token.lemma_).lower()
                if (token.lemma_).lower() not in unique_words:
                    unique_words.append((token.lemma_).lower())
                tk=token.pos_
                if tk not in unique_word_types:
                    unique_word_types.append(tk)
        lists.append(string)
    return lists

def calculateFscore():
    vectorizer = CountVectorizer()
    trainX = vectorizer.fit_transform(preprocessTrainData())
    testX = vectorizer.transform(preprocessTestData())
    trainY=np.vstack((np.zeros([594,1],dtype=int),np.ones([600,1],dtype=int)))
    testY=np.vstack((np.zeros([396,1],dtype=int),np.ones([399,1],dtype=int)))
    naivebayes=MultinomialNB()
    naivebayes.fit(trainX,trainY.ravel())
    MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
    testY1=[]
    for i in testY:
        testY1.append(i[0])
    print("Accuracy",accuracy_score(naivebayes.predict(testX),testY1))
    print("F1 score",f1_score(testY,naivebayes.predict(testX),average="binary"))
    
trainingData()   
testingData()
calculateFscore()
endTime = datetime.now()
print(endTime)
print("Total time for execution:",endTime-startTime)