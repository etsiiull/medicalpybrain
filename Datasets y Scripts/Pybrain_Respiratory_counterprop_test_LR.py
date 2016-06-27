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
import csv

#Leer las bases de datos 
patternTrain = np.loadtxt("RespiratorySystemCancerKohonenTrain.csv", dtype=float, delimiter=',')
patternValid = np.loadtxt("RespiratorySystemCancerKohonenValid.csv", dtype=float, delimiter=',')
patternTest = np.loadtxt("RespiratorySystemCancerKohonenTest.csv", dtype=float, delimiter=',')

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
	
resultados = np.zeros((30,9))
myLearningRate = 0.0001
counterOut = 0
while(myLearningRate < 0.09):
	counter = 0
	while(counter < 5):
		#Crear y entrenar el mapa autoorganizado
		neuronas = 15
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
	
		#Crear la red para el backprop
		#myLearningRate = 0.05
		myMomentum = 0.5
		net = buildNetwork(neuronas**2, 1, outclass=SigmoidLayer, bias=True)

		#Crear el trainer y entrenarlo con los DS
		trainer = BackpropTrainer(net, kohonenDS, learningrate=myLearningRate, momentum=myMomentum)
		trainError = trainer.trainUntilConvergence(verbose=True, trainingData=kohonenDS, validationData=kohonenValidDS, maxEpochs=100)
	
		trainResult = net.activateOnDataset(kohonenValidDS)
		print("Train Result: %s" % trainResult)

		#Calcular accuracy, sensibilidad y especificidad
		patResult = -1
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
			
		print("Learning Rate: %1.4f" % myLearningRate)
		print("Iteracion: %d" % counter)
		print("\n")
		
		print("Positivo: %d" % positivo)
		print("Negativo: %d" % negativo)
		print("Falso Positivo: %d" % falsoPositivo)
		print("Falso Negativo: %d" % falsoNegativo)
		print("\n")
		
		minError = 0.07
		#C?lculo del numero de iteraciones antes de llegar al 0,07 de error.
		iteracionesTrain = -1
		iteracionesValid = -1
		for i in range(len(trainError[0])):
			if ((trainError[0][i] < minError) and (iteracionesTrain == -1)):
				iteracionesTrain = i
			if ((trainError[1][i] < minError) and (iteracionesValid == -1)):
				iteracionesValid = i

		accuracy = (positivo + negativo) / numPatTest
		sensibilidad = positivo / (positivo + falsoNegativo)
		especificidad = negativo /(negativo + falsoPositivo)

		print("Accuracy : %1.3f" % accuracy)
		print("Sensibilidad: %1.3f" % sensibilidad)
		print("Especificidad: %1.3f" % especificidad)
		print("Iteraciones de entrenamiento: %d" % iteracionesTrain)
		print("Iteraciones de validaci?n: %d" % iteracionesValid)
	
		result_array = [positivo, negativo, falsoPositivo, falsoNegativo, accuracy, sensibilidad, especificidad, iteracionesTrain, iteracionesValid]
		resultados[counterOut] = result_array
		
		counter += 1
		counterOut += 1
		
	if(myLearningRate == 0.0001):
		myLearningRate = 0.0005
	elif(myLearningRate == 0.0005):
		myLearningRate = 0.001
	elif(myLearningRate == 0.001):
		myLearningRate = 0.005
	elif(myLearningRate == 0.005):
		myLearningRate = 0.01
	elif(myLearningRate == 0.01):
		myLearningRate = 0.05
	elif(myLearningRate == 0.05):
		myLearningRate = 0.1
	
with open('resultsRespiratoryLRCounterprop.csv', 'w', newline='') as fp:
	writer = csv.writer(fp, delimiter=',')
	writer.writerows(resultados)
