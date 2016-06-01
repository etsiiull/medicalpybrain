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

resultados = np.zeros((45,9))
myMomentum = 0.1
counterOut = 0
minError = 0.13
while(myMomentum < 0.9):
	counter = 0
	while(counter < 5):	
		#Crear red con una capa oculta
		numHiddenNodes = 50
		myLearningRate = 0.005
		#myMomentum = 0.1
		myEpochs = 150
		net = buildNetwork(numColsTrain-1, numHiddenNodes, 2, bias=True)

		#Crear el trainer y hacer enternar el DS
		trainer = BackpropTrainer(net, trainDS, learningrate=myLearningRate, momentum=myMomentum)
		trainError = trainer.trainUntilConvergence(verbose=True, trainingData=trainDS, validationData=validDS, maxEpochs=myEpochs)

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
		
		print("Valor de Momentum: %1.1f" % myMomentum)
		print("Iteracion: %d" % counter)
		print("\n")
		
		print("Positivo: %d" % positivo)
		print("Negativo: %d" % negativo)
		print("Falso Positivo: %d" % falsoPositivo)
		print("Falso Negativo: %d" % falsoNegativo)
		print("\n")

		positivoTotal = positivo + falsoNegativo
		negativoTotal = negativo + falsoPositivo

		percentPositivo = positivo / positivoTotal * 100
		percentNegativo = negativo / negativoTotal * 100
		percentFalsoPositivo = falsoPositivo / negativoTotal * 100
		percentFalsoNegativo = falsoNegativo / positivoTotal * 100
		accuracy = ((positivo + negativo) / numPatValid) * 100
		recall = (positivo / positivoTotal) * 100
		if(positivo == 0 and falsoPositivo == 0):
			precision = 0
		else:
			precision = (positivo / (positivo + falsoPositivo)) * 100
		
		#Cálculo del numero de iteraciones antes de llegar al 0,13 de error.
		iteracionesTrain = -1
		iteracionesValid = -1
		for i in range(len(trainError[0])):
			if ((trainError[0][i] < minError) and (iteracionesTrain == -1)):
				iteracionesTrain = i
			if ((trainError[1][i] < minError) and (iteracionesValid == -1)):
				iteracionesValid = i		
				
		print("Porcentaje de aciertos positivos: %3.2f%%" % percentPositivo)
		print("Porcentaje de falsos negativos: %3.2f%%" % percentFalsoNegativo)
		print("Porcentaje de aciertos negativos: %3.2f%%" % percentNegativo)
		print("Porcentaje de falsos positivos: %3.2f%%" % percentFalsoPositivo)
		print("\n")

		print("Accuracy: %3.2f%%" % accuracy)
		print("Recall: %3.2f%%" % recall)
		print("Precision: %3.2f%%" % precision)
		print("\n")
		
		print("Iteraciones de entrenamiento: %d" % iteracionesTrain)
		print("Iteraciones de validación: %d" % iteracionesValid)
		
		result_array = [positivo, negativo, falsoPositivo, falsoNegativo, accuracy, recall, precision, iteracionesTrain, iteracionesValid]
		resultados[counterOut] = result_array
		
		counterOut = counterOut + 1
		counter = counter + 1
	
	myMomentum = myMomentum + 0.1
	
with open('resultsRespiratoryMomentum.csv', 'w', newline='') as fp:
	writer = csv.writer(fp, delimiter=',')
	writer.writerows(resultados)