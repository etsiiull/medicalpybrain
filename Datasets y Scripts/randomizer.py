import numpy as np

tabla = np.loadtxt("RespiratorySystemCancerNormalize.csv", dtype=float, delimiter=',')

np.random.seed(12345)
np.random.shuffle(tabla)
rows,cols = tabla.shape

traintabla = tabla[:rows/2,:]
validtabla = tabla[rows/2:3*(rows/4),:]
testtabla = tabla[3*(rows/4):,:]

np.savetxt('RespiratoryCancerNormalizeTrain.csv', traintabla)
np.savetxt('RespiratoryCancerNormalizeValid.csv', validtabla)
np.savetxt('RespiratoryCancerNormalizeTest.csv', testtabla)



