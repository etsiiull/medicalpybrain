import numpy as np

tabla = np.loadtxt("BreastCancerPreprocessed.csv", dtype=float, delimiter=',')

np.random.seed(12345)
np.random.shuffle(tabla)
rows,cols = tabla.shape

traintabla = tabla[:rows/2,:]
validtabla = tabla[rows/2:3*(rows/4),:]
testtabla = tabla[3*(rows/4):,:]

np.savetxt('BreastCancerPreprocessedTrain.csv', traintabla)
np.savetxt('BreastCancerPreprocessedValid.csv', validtabla)
np.savetxt('BreastCancerPreprocessedTest.csv', testtabla)



