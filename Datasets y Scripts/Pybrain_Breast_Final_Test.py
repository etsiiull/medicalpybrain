#Imports
from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
import numpy as np
import pylab as pl
import math as ma
import csv

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

resultados = np.zeros((10,6))
myMaxEpochs = 150
counterOut = 0
while(counterOut < 10):
		#Crear red con una capa oculta
		numHiddenNodes = 50
		myLearningRate = 0.005
		myMomentum = 0.1
		net = buildNetwork(numColsTrain-1, numHiddenNodes, 2, bias=True)

		#Crear el trainer y hacer enternar el DS
		trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
		trainError = trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=myMaxEpochs)

		#Obtener porcentajes
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
		
		print("Iteracion: %d" % counterOut)
		print("\n")
		
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
		
		result_array = [positivo, negativo, falsoPositivo, falsoNegativo, sensibilidad, especificidad]
		resultados[counterOut] = result_array
		
		counterOut = counterOut + 1
		
		NetworkWriter.writeToFile(net, "%d.xml" % counterOut)
	
