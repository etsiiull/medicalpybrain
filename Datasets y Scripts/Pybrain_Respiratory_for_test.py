#Imports
from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import numpy as np
import pylab as pl
import math as ma

#Read training sets
patternTrain = np.loadtxt("RespiratorySystemCancerPreprocessedTrain.csv")
patternValid = np.loadtxt("RespiratorySystemCancerPreprocessedValid.csv")
patternTest = np.loadtxt("RespiratorySystemCancerPreprocessedTest.csv")

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
	
counter = 0
while(counter < 5):	
	#Crear red con una capa oculta
	numHiddenNodes = 100	
	myLearningRate = 0.0001
	myMomentum = 0.1
	net = buildNetwork(numColsTrain-1, numHiddenNodes, 2, bias=True)

	#Crear el trainer y hacer enternar el DS
	trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
	trainError = trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=100)

	#Plot training and validation errors
	#pl.plot(trainError[0], label='Train Error')
	#pl.plot(trainError[1], label='Valid Error')
	#pl.xlabel('Epoch num')
	#pl.ylabel('Error')
	#pl.legend(loc='upper right')
	#pl.show()

	results = net.activateOnDataset(validDS)

	patResult = -1
	positivo = 0
	negativo = 0
	falsoPositivo = 0
	falsoNegativo = 0

	for i in range(numPatValid):
		if max(results[i]) == results[i, 0]:
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
		
	print(positivo)
	print(negativo)
	print(falsoPositivo)
	print(falsoNegativo)

	positivoTotal = positivo + falsoNegativo
	negativoTotal = negativo + falsoPositivo

	percentPositivo = positivo / positivoTotal * 100
	percentNegativo = negativo / negativoTotal * 100
	percentFalsoPositivo = falsoPositivo / negativoTotal * 100
	percentFalsoNegativo = falsoNegativo / positivoTotal * 100
	percentTotal = ((positivo + negativo) / numPatValid) * 100

	print("Porcentaje de aciertos positivos: %3.2f%%" % percentPositivo)
	print("Porcentaje de falsos negativos: %3.2f%%" % percentFalsoNegativo)
	print("Porcentaje de aciertos negativos: %3.2f%%" % percentNegativo)
	print("Porcentaje de falsos positivos: %3.2f%%" % percentFalsoPositivo)
	print("Porcentaje total de acierto: %3.2f%%" % percentTotal)
	
	counter = counter + 1