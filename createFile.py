import numpy as np

np.random.seed( 1212 )

n = 120

oFile = open( 'data_1.csv', 'w' )

for i in range( n ):
	line = '%d,%f,%f,%f\n' % ( i, np.random.rand(), np.random.rand(), np.random.rand() )
	oFile.write( line )
oFile.close()
