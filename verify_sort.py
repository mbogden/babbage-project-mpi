# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
from sys import argv, exit
import numpy as np
import pandas as pd
from hashlib import md5

# For troubleshooting
import time

# Global variable for now
printAll = True
printAll = False

size = int( argv[3] )
 
print('input data: %s' % argv[1])

# Read in csv data file passed in command line
pd_in = pd.read_csv( argv[1], header=None ).to_numpy( dtype=np.float32, copy=True )
send_data = np.empty( pd_in.shape, dtype=np.float32 )
send_data[:,:] = pd_in[:,:]
del pd_in

sort_col = int( argv[2] )

def bin_my_array( my_array ):

	# Edges of bins known to all nodes
	bins = np.linspace( 0, 1, size +1 )

	# Create python list to seperate data
	data_to_nodes = []

	# Loop through bin edges and 
	# Extract rows of data between bin edges
	for i in range( size ):
		data_to_nodes.append( my_array[(my_array[:,sort_col] < bins[i+1]) & (my_array[:,sort_col] >= bins[i])] )

	return data_to_nodes

# Bin my array to subarray to send to each node
data_to_nodes = bin_my_array( send_data )



for i,my_array in enumerate(data_to_nodes):

		my_array = my_array[ np.argsort( my_array[ :, sort_col ] ) ]

		# For verification, Extract x random rows for hash subsampling
		rand_n = 5
		np.random.seed( 1212 )
		my_str = ''
		for j in range(rand_n):
				rand_i = np.random.randint( my_array.shape[0] )
				rand_j = np.random.randint( my_array.shape[1] )
				my_str += '%13e'% my_array[ rand_i, rand_j ]
		my_md5 = md5( my_str.encode() ).hexdigest()

		print("%d : %s"%(i, my_md5))
