#Imports
from __future__ import division
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.svmunit import SVMUnit
from pybrain.supervised.trainers.svmtrainer import SVMTrainer
from pybrain.utilities import percentError
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
trainDS = ClassificationDataSet(numColsTrain-1, nb_classes=2, class_labels=['Not_Cancer', 'Cancer'])
for i in range(numPatTrain):
	trainDS.appendLinked(patternTrainInput[i], patternTrain[i, 0])
	
validDS = ClassificationDataSet(numColsTrain-1, nb_classes=2, class_labels=['Not_Cancer', 'Cancer'])
for i in range(numPatValid):
	validDS.appendLinked(patternValidInput[i], patternValid[i, 0])
	
testDS = ClassificationDataSet(numColsTrain-1, nb_classes=2, class_labels=['Not_Cancer', 'Cancer'])
for i in range(numPatTest):
	testDS.appendLinked(patternTestInput[i], patternTest[i, 0])

#Crear la SVM y el trainer
svm = SVMUnit()
trainer = SVMTrainer(svm, trainDS)

#Parámetros de la SVM
myLog2C=0.
myLog2g=1.1

#Entrenar la red
trainer.train(log2g=myLog2g, log2C=myLog2C)

#
trnresult = percentError( svm.activateOnDataset(trndata), trndata['target'] )
tstresult = percentError( svm.activateOnDataset(tstdata), tstdata['target'] )
print("sigma: %7g,  C: %7g,  train error: %5.2f%%,  test error: %5.2f%%" % (2.0**myLog2g, 2.0**myLog2C, trnresult, tstresult))
	
