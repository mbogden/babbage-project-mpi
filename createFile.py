import numpy as np

n = 12

oFile = open( 'test_data.csv', 'w' )

for i in range( 12 ):
	line = '%d,%f,%f,%f\n' % ( i, np.random.rand(), np.random.rand(), np.random.rand() )
	oFile.write( line )
oFile.close()
