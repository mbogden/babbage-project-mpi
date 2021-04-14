# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
from sys import argv, exit
import numpy as np
import pandas as pd
from hashlib import md5
from mpi4py import MPI, rc
rc.recv_mprobe = False

# For troubleshooting
import time

# Global variable for now
printAll = True
#printAll = False
 
# Grab useful things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print( 'I am rank %d of %d' % ( rank, size ) )
 
# Create empty data for later broadcast
send_data = None
size_col = np.zeros( 2, dtype=np.uint64 )

if rank == 0:
	print('input data: %s' % argv[1])

	# Read in csv data file passed in command line
	pd_in = pd.read_csv( argv[1], header=None ).to_numpy( dtype=np.float32, copy=True )

	# We want row major arrays to function properly with MPI Scsatter.  Pandas reads column major (Or vice versa)
	# This swaps to row major, we instantiate a numpy array which default to row major, then copy the data
	# If memory is a concern, we can create, copy, delete in chunks. 
	# Or find the mythical function that does it for us.
	send_data = np.empty( pd_in.shape, dtype=np.float32 )
	send_data[:,:] = pd_in[:,:]
	del pd_in

	# Old numpy reading file
	# np_send_data = np.loadtxt( argv[1], delimiter=',', dtype=np.float32 )

	if printAll: 
		#print('Node 0 read in numpy: \n', np_send_data.flags,'\n', np_send_data, '\n')
		print('Node 0 read in pandas: \n', send_data.flags,'\n', send_data, '\n')
	
	# Rank 0 calculates how much data is going to each node, and which array to sort by
	buffer_constant = 2.0
	n_per_node = round( buffer_constant * float(send_data.shape[0]) / float(size) )

	sort_col = int( argv[2] )
	
	# Create 2 integers to pass to nodes before passing giant data
	size_col[0] = n_per_node
	size_col[1] = sort_col

	if printAll: 
			print( 'Size of input data: ', send_data.shape[0] )
			print( 'N per node ', size_col[0] )
			print( 'Sort by col: ', size_col[1] )


# Rank 0 will do a "one to all" call saying how much data to expect, and which array to sort by
size_col = comm.bcast( size_col, root = 0 )

if printAll:

	# Syncronizing and offsetting time organized the print statements
	comm.Barrier()
	time.sleep( rank*0.25 )
	print( 'Rank %d received size_col: '% rank, size_col)

# Save as seperate named variables
my_n = size_col[0]
sort_col = size_col[1]

# Create a recieving buffer with np.nan as placeholder
my_array = np.empty( ( my_n, 4 ), dtype=np.float32 )
my_array[:] = np.nan

# Rank 0 is scattering the initial data to all nodes
#send_data = np.transpose( send_data )
comm.Scatter( send_data, my_array, root=0 )

# delete in data if rank 0
if rank == 0:
	del send_data

if printAll:
	comm.Barrier()
	time.sleep( rank*0.25 )
	print( 'Rank %d received my_array: \n'%rank, my_array )

def bin_my_array( my_array ):

	# Edges of bins known to all nodes
	bins = np.linspace( 0, 1, size +1 )

	# Create python list to seperate data
	data_to_nodes = []

	# Loop through bin edges and 
	# Extract rows of data between bin edges
	for i in range( size ):
		data_to_nodes.append( my_array[(my_array[:,sort_col] < bins[i+1]) & (my_array[:,sort_col] >= bins[i])] )
	

	if printAll:
		comm.Barrier()
		time.sleep( rank*0.25 )

		print( 'Rank %d array:'%rank)
		for i in range( size ):				
			print( '\nRank %d -> %d' % ( rank, i ) )
			print( data_to_nodes[i] )

	return data_to_nodes

# Bin my array to subarray to send to each node
data_to_nodes = bin_my_array( my_array )

# Create integer array of how many items are going to each node
count_to_nodes = np.zeros( size, dtype=np.uint64 )

# Send max nodes instead
for i in range( size ):
	count_to_nodes[i] = len( data_to_nodes[i] )

to_max = int( np.amax( count_to_nodes ) )

count_to_nodes[:] = to_max

if printAll:
		comm.Barrier()
		time.sleep( rank*0.25 )
		print( 'Rank %d count_to_nodes: ' % rank, count_to_nodes )

# Make all to all mpi call passing how many data items are being send to each
count_from_nodes = comm.alltoall( count_to_nodes )
max_n = int( np.amax( count_from_nodes ) )

if printAll:
		comm.Barrier()
		time.sleep( rank*0.25 )
		print( 'Rank %d count_from_nodes: '%rank, count_from_nodes )

send_all_array = np.empty( ( size, max_n, 4 ), dtype=np.float32 )
send_all_array[:] = np.nan

for i, send_data in enumerate( data_to_nodes ):
	if send_data.size <= 0:
		continue
	send_all_array[ i, 0 : send_data.shape[0], : ] = send_data

#count_from_nodes = comm.alltoall( count_to_nodes )
recv_all_array = comm.alltoall( send_all_array )

if printAll: 
		comm.Barrier()
		time.sleep( rank*0.25 )
		print( 'Rank %d sending all_to_all matrix' % rank)
		for i in range( size ):
			print( send_all_array[i] )

# Reshape in long list with 4 columns
recv_all_array = np.reshape( recv_all_array, ( -1, 4 ) )

# Remove NaN rows
my_array = recv_all_array[~np.isnan(recv_all_array).any(axis=1)]

if printAll:
		comm.Barrier()
		time.sleep( rank*0.5 )
		print( 'Rank %d will sort my_array:\n' % rank, my_array )


# Sort my_array by column
my_array = my_array[ np.argsort( my_array[:, sort_col] ) ]

if printAll:
		comm.Barrier()
		time.sleep( rank*0.25 )
		print( "Rank %d sorted array!: \n "%rank, my_array )

# For verification, Extract x random rows for hash subsampling
rand_n = 5
np.random.seed( 1212 )
my_str = ''
for i in range(rand_n):
		rand_i = np.random.randint( my_array.shape[0] )
		rand_j = np.random.randint( my_array.shape[1] )
		my_str += '%13e'% my_array[ rand_i, rand_j ]
my_md5 = md5( my_str.encode() ).hexdigest()

comm.Barrier()
time.sleep( rank*0.25 )
print("%d : %s"%(rank, my_md5))
