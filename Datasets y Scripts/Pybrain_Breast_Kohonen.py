#Imports
from __future__ import division
from scipy import random
from pybrain.structure.modules import KohonenMap
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import pylab as pl
import math as ma

#Leer las bases de datos 
patternTrain = np.loadtxt("BreastCancerPreprocessedTrain.csv", dtype=float, delimiter=',')
patternValid = np.loadtxt("BreastCancerPreprocessedValid.csv", dtype=float, delimiter=',')
patternTest = np.loadtxt("BreastCancerPreprocessedTest.csv", dtype=float, delimiter=',')

#Conseguir el numero de filas y columnas
numPatTrain, numColsTrain = patternTrain.shape
numPatValid, numColsValid = patternValid.shape
numPatTest, numColsTest = patternTest.shape

#Generar el input
patternTrainInput = patternTrain[:, 1:numColsTrain]
patternValidInput = patternValid[:, 1:numColsValid]
patternTestInput = patternTest[:, 1:numColsTest]

#Generar salidas deseadas 
patternTrainTarget = np.zeros([numPatTrain, 2])
patternValidTarget = np.zeros([numPatValid, 2])
patternTestTarget = np.zeros([numPatTest, 2])

#Crear los dataset supervisados
trainDS = SupervisedDataSet(numColsTrain-1, 2)
for i in range(numPatTrain):
	patternTrainTarget[i, patternTrain[i, 0]] = 1.0
	trainDS.addSample(patternTrainInput[i], patternTrainTarget[i])
	
validDS = SupervisedDataSet(numColsValid-1, 2)
for i in range(numPatValid):
	patternValidTarget[i, patternValid[i, 0]] = 1.0
	validDS.addSample(patternValidInput[i], patternValidTarget[i])
	
testDS = SupervisedDataSet(numColsTest-1, 2)
for i in range(numPatTest):
	patternTestTarget[i, patternTest[i, 0]] = 1.0
	testDS.addSample(patternTestInput[i], patternTestTarget[i])
	
som = KohonenMap(numColsTrain-1, 50, outputFullMap=False)
print(numColsValid-1)

for i in range(numPatTrain):
    # one forward and one backward (training) pass
    som.activate(patternTrainInput[i])
    som.backward()

result = som.activate(patternValidInput[0])
print(float(result[0]))

#Crear red sin capa oculta
myLearningRate = 0.0001
myMomentum = 0.1
net = buildNetwork(numColsTrain-1, 2)

#Crear el trainer y hacer enternar el DS
trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=100)

results = net.activate(som.activate(patternValidInput[0]))
print(results)
