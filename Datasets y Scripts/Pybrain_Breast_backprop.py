#Imports
from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import pylab as pl
import math as ma

#Leer las bases de datos 
patternTrain = np.loadtxt("BreastCancerNormalizeTrain.csv", dtype=float, delimiter=',')
patternValid = np.loadtxt("BreastCancerNormalizeValid.csv", dtype=float, delimiter=',')
patternTest = np.loadtxt("BreastCancerNormalizeTest.csv", dtype=float, delimiter=',')

#Conseguir el numero de filas y columnas
numPatTrain, numColsTrain = patternTrain.shape
numPatValid, numColsValid = patternValid.shape
numPatTest, numColsTest = patternTest.shape

#Generar el input
patternTrainInput = patternTrain[:, 1:numColsTrain]
patternValidInput = patternValid[:, 1:numColsValid]
patternTestInput = patternTest[:, 1:numColsTest]

#Generar salidas deseadas 
patternTrainTarget = np.zeros([numPatTrain])
patternValidTarget = np.zeros([numPatValid])
patternTestTarget = np.zeros([numPatTest])

#Crear los dataset supervisados
trainDS = SupervisedDataSet(numColsTrain-1, 1)
for i in range(numPatTrain):
	patternTrainTarget[i] = patternTrain[i, 0]
	trainDS.addSample(patternTrainInput[i], patternTrainTarget[i])
	
validDS = SupervisedDataSet(numColsValid-1, 1)
for i in range(numPatValid):
	patternValidTarget[i] = patternValid[i, 0]
	validDS.addSample(patternValidInput[i], patternValidTarget[i])
	
testDS = SupervisedDataSet(numColsTest-1, 1)
for i in range(numPatTest):
	patternTestTarget[i] = patternTest[i, 0]
	testDS.addSample(patternTestInput[i], patternTestTarget[i])

#Crear red con una capa oculta
numHiddenNodes = 50
myLearningRate = 0.005
myMomentum = 0.5
net = buildNetwork(numColsTrain-1, numHiddenNodes, 1, outclass=SigmoidLayer, bias=True)

#Crear el trainer y hacer enternar el DS
trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
trainError = trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=100)

#Crear la gráfica con los errores de validación y entrenamiento
pl.plot(trainError[0], label='Train Error')
pl.plot(trainError[1], label='Valid Error')
pl.xlabel('Epoch num')
pl.ylabel('Error')
pl.legend(loc='upper right')
pl.show()

#Obtener porcentajes
trainResult = net.activateOnDataset(validDS)
print("Results: %s" % trainResult)

patResult = 0
positivo = 0
negativo = 0
falsoPositivo = 0
falsoNegativo = 0

threshold = 0.45
for i in range(numPatValid):
	if trainResult[i] <= threshold:
		patResult = 0
	else:
		patResult = 1
		
	if (patternValid[i, 0] == 1 and patternValid[i, 0] == patResult):
		positivo = positivo + 1
	elif (patternValid[i, 0] == 0 and patternValid[i, 0] == patResult):
		negativo = negativo + 1
	elif (patternValid[i, 0] == 1 and patternValid[i, 0] != patResult):
		falsoNegativo = falsoNegativo + 1
	elif (patternValid[i, 0] == 0 and patternValid[i, 0] != patResult):
		falsoPositivo = falsoPositivo + 1
		
print("Positivo: %d" % positivo)
print("Negativo: %d" % negativo)
print("Falso Positivo: %d" % falsoPositivo)
print("Falso Negativo: %d" % falsoNegativo)
print("\n")

accuracy = (positivo + negativo) / numPatTest
sensibilidad = positivo / (positivo + falsoNegativo)
especificidad = negativo /(negativo + falsoPositivo)

print("Accuracy : %1.3f" % accuracy)
print("Sensibilidad: %1.3f" % sensibilidad)
print("Especificidad: %1.3f" % especificidad)