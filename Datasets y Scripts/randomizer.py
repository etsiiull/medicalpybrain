import numpy as np

tabla = np.loadtxt("RespiratorySystemCancerKohonen.csv", dtype=float, delimiter=',')

np.random.seed(12345)
np.random.shuffle(tabla)
rows,cols = tabla.shape

traintabla = tabla[:rows/2,:]
validtabla = tabla[rows/2:3*(rows/4),:]
testtabla = tabla[3*(rows/4):,:]

np.savetxt('RespiratorySystemCancerKohonenTrain.csv', traintabla)
np.savetxt('RespiratorySystemCancerKohonenValid.csv', validtabla)
np.savetxt('RespiratorySystemCancerKohonenTest.csv', testtabla)



