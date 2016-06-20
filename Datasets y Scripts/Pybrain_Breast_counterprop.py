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

if numColsTrain == numColsValid == numColsTest:
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

neuronas = 10	
	
#Crear y entrenar el mapa autoorganizado
som = KohonenMap(numColsTrain-1, neuronas)
	
#Entrenar el mapa	
for i in range(numPatTrain):
	som.activate(patternTrainInput[i])
	som.backward()	
	
#Crear el dataset de entrenamiento de backprop con resultados del mapa
input = np.zeros([numPatTrain,neuronas**2])
for i in range(numPatTrain):
	tmp = som.activate(patternTrainInput[i])
	print(tmp)
	inputTMP = np.zeros(neuronas**2)
	inputTMP[(neuronas*tmp[0]) + tmp[1]] = 1.0
	print(inputTMP)
	input[i] = inputTMP
	
kohonenDS = SupervisedDataSet(neuronas**2, 2)
for i in range(numPatTrain):
	kohonenDS.addSample(input[i], patternTrainTarget[i])
	
#Crear el dataset de validacion de backprop con resultados del mapa
input = np.zeros([numPatValid,neuronas**2])
for i in range(numPatValid):
	tmp = som.activate(patternValidInput[i])
	print(tmp)
	inputTMP = np.zeros(neuronas**2)
	inputTMP[(neuronas*tmp[0]) + tmp[1]] = 1.0
	print(inputTMP)
	input[i] = inputTMP
	
kohonenValidDS = SupervisedDataSet(neuronas**2, 2)
for i in range(numPatValid):
	kohonenValidDS.addSample(input[i], patternValidTarget[i])
	
#Crear la red para el backprop
myLearningRate = 0.005
myMomentum = 0.7
net = buildNetwork(neuronas**2, 100, 2, bias=True)

#Crear el trainer y entrenarlo con los DS
trainer = BackpropTrainer(net, kohonenDS, learningrate=myLearningRate, momentum=myMomentum)
trainer.trainUntilConvergence(verbose=True, trainingData=kohonenDS, validationData=kohonenValidDS, maxEpochs=100)

#Resultados de validación
results = np.zeros([numPatValid,neuronas**2])
for i in range(numPatValid):
	tmp = som.activate(patternValidInput[i])
	inputTMP = np.zeros(neuronas**2)
	inputTMP[neuronas*tmp[0] + tmp[1]] = 1.0
	results[i] = inputTMP
	
kohonenResultDS = SupervisedDataSet(neuronas**2, 2)
for i in range(numPatValid):
	kohonenResultDS.addSample(results[i], patternValidTarget[i])
	
trainResult = net.activateOnDataset(kohonenResultDS)
print("Train Result: %s" % trainResult)

#Calcular accuracy, sensibilidad y especificidad
patResult = -1
positivo = 0
negativo = 0
falsoPositivo = 0
falsoNegativo = 0

for i in range(numPatValid):
	if max(trainResult[i]) == trainResult[i, 0]:
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
