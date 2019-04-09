# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

"""

import numpy as np
from ClassifySpam import aucCV, aucTest
#from classifySpamLr import aucCV, aucTest

nRuns = 10

trainData = np.loadtxt('spamTrain.csv',delimiter=',')
testData = np.loadtxt('spamTest.csv',delimiter=',')

aucCVRun = np.zeros(nRuns)
aucTestRun = np.zeros(nRuns)

for run in range(nRuns):
    print("Run ", run)
    
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(trainData)[0])
    np.random.shuffle(shuffleIndex)
    trainData = trainData[shuffleIndex,:]
    trainFeatures = trainData[:,:-1]
    trainLabels = trainData[:,-1]
    aucCVRun[run] = np.mean(aucCV(trainFeatures,trainLabels))
    
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(testData)[0])
    np.random.shuffle(shuffleIndex)
    testData = testData[shuffleIndex,:]
    testFeatures = testData[:,:-1]
    testLabels = testData[:,-1]
    aucTestRun[run] = aucTest(trainFeatures,trainLabels,testFeatures,
                              testLabels)

print("10-fold cross-validation mean AUC: ", np.mean(aucCVRun))
print("Test set AUC: ", np.mean(aucTestRun))
