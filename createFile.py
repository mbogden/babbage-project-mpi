import numpy as np
from sys import argv
np.random.seed( 1212 )

n = int( argv[1] )
saveFile = argv[2]

oFile = open( saveFile, 'w' )

for i in range( n ):
	line = '%d,%f,%f,%f\n' % ( i, np.random.rand(), np.random.rand(), np.random.rand() )
	oFile.write( line )
oFile.close()
