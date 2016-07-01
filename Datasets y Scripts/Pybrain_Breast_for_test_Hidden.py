#Imports
from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import numpy as np
import pylab as pl
import math as ma
import csv

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

resultados = np.zeros((100,7))
numHiddenNodes = 5
while(numHiddenNodes < 101):
	counter = 0
	while(counter < 5):	
		#Crear red con una capa oculta
		#numHiddenNodes = 25
		myLearningRate = 0.0001
		myMomentum = 0.1
		net = buildNetwork(numColsTrain-1, numHiddenNodes, 1, bias=True)

		#Crear el trainer y hacer enternar el DS
		trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
		trainError = trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=100)

		#Obtener porcentajes
		results = net.activateOnDataset(validDS)

		patResult = -1
		positivo = 0
		negativo = 0
		falsoPositivo = 0
		falsoNegativo = 0

		threshold = 0.45
		for i in range(numPatValid):
			if results[i] <= threshold:
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
			
		print("Nº de Neuronas ocultas: %d" % numHiddenNodes)
		print("Iteracion: %d" % counter)
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
		
		result_array = [positivo, negativo, falsoPositivo, falsoNegativo, accuracy, sensibilidad, especificidad]
		resultados[numHiddenNodes - 5 + counter] = result_array
		
		counter = counter + 1
	
	numHiddenNodes = numHiddenNodes + 5
	
with open('resultsBreastHidden.csv', 'w', newline='') as fp:
	writer = csv.writer(fp, delimiter=',')
	writer.writerows(resultados)