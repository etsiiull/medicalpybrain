#Imports
from __future__ import division
from scipy import random
from pybrain.structure.modules import KohonenMap, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import pylab as pl
import math as ma

#Leer las bases de datos 
patternTrain = np.loadtxt("MelanomaNormalizeTrain.csv", dtype=float, delimiter=',')
patternValid = np.loadtxt("MelanomaNormalizeValid.csv", dtype=float, delimiter=',')
patternTest = np.loadtxt("MelanomaNormalizeTest.csv", dtype=float, delimiter=',')

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

for i in range(numPatTrain):
	patternTrainTarget[i] = patternTrain[i, 0]
	
for i in range(numPatValid):
	patternValidTarget[i] = patternValid[i, 0]
	
for i in range(numPatTest):
	patternTestTarget[i] = patternTest[i, 0]

counterOut = 0
while(counterOut < 10):
	neuronas = 11
	#Crear y entrenar el mapa autoorganizado
	som = KohonenMap(numColsTrain-1, neuronas)
	
	#Entrenar el mapa	
	for i in range(numPatTrain):
		som.activate(patternTrainInput[i])
		som.backward()
	
	#Crear el dataset de entrenamiento de backprop con resultados del mapa
	input = np.zeros([numPatTrain,neuronas**2])
	distMatrix = np.zeros([neuronas, neuronas])
	for i in range(numPatTrain):
		tmp = som.activate(patternTrainInput[i])
		distMatrix[tmp[0], tmp[1]] += 1
		inputTMP = np.zeros(neuronas**2)
		inputTMP[(neuronas*tmp[0]) + tmp[1]] = 1.0
		input[i] = inputTMP
		
	kohonenDS = SupervisedDataSet(neuronas**2, 1)
	for i in range(numPatTrain):
		kohonenDS.addSample(input[i], patternTrainTarget[i])

	#Crear el dataset de validacion de backprop con resultados del mapa
	input = np.zeros([numPatValid,neuronas**2])
	distMatrix = np.zeros([neuronas, neuronas])
	for i in range(numPatValid):
		tmp = som.activate(patternValidInput[i])
		distMatrix[tmp[0], tmp[1]] += 1
		inputTMP = np.zeros(neuronas**2)
		inputTMP[(neuronas*tmp[0]) + tmp[1]] = 1.0
		input[i] = inputTMP
	
	kohonenValidDS = SupervisedDataSet(neuronas**2, 1)
	for i in range(numPatValid):
		kohonenValidDS.addSample(input[i], patternValidTarget[i])
		
	#Crear el dataset de test de backprop con resultados del mapa
	input = np.zeros([numPatTest,neuronas**2])
	distMatrix = np.zeros([neuronas, neuronas])
	for i in range(numPatValid):
		tmp = som.activate(patternTestInput[i])
		distMatrix[tmp[0], tmp[1]] += 1
		inputTMP = np.zeros(neuronas**2)
		inputTMP[(neuronas*tmp[0]) + tmp[1]] = 1.0
		input[i] = inputTMP
		
	kohonenTestDS = SupervisedDataSet(neuronas**2, 1)
	for i in range(numPatTest):
		kohonenTestDS.addSample(input[i], patternTestTarget[i])
	
	#Crear la red para el backprop
	myLearningRate = 0.05
	myMomentum = 0.9
	net = buildNetwork(neuronas**2, 1, outclass=SigmoidLayer, bias=True)

	#Crear el trainer y entrenarlo con los DS
	trainer = BackpropTrainer(net, kohonenDS, learningrate=myLearningRate, momentum=myMomentum)
	trainError = trainer.trainUntilConvergence(verbose=True, trainingData=kohonenDS, validationData=kohonenValidDS, maxEpochs=100)
	
	trainResult = net.activateOnDataset(kohonenTestDS)

	#Calcular accuracy, sensibilidad y especificidad
	patResult = -1
	positivo = 0
	negativo = 0
	falsoPositivo = 0
	falsoNegativo = 0

	threshold = 0.45
	for i in range(numPatTest):
		if trainResult[i] <= threshold:
			patResult = 0
		else:
			patResult = 1
			
		if (patternTest[i, 0] == 1 and patternTest[i, 0] == patResult):
			positivo = positivo + 1
		elif (patternTest[i, 0] == 0 and patternTest[i, 0] == patResult):
			negativo = negativo + 1
		elif (patternTest[i, 0] == 1 and patternTest[i, 0] != patResult):
			falsoNegativo = falsoNegativo + 1
		elif (patternTest[i, 0] == 0 and patternTest[i, 0] != patResult):
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
		
	counterOut += 1

