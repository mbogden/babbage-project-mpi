# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
from sys import argv, exit
import numpy as np
import pandas as pd
from mpi4py import MPI

# For troubleshooting
import time

# Global variable for now
 
# Grab useful things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print( 'I am rank %d of %d' % ( rank, size ) )
 
# Create data to broadcast
send_data = None
pd_send_data = None
size_col = np.zeros( 2, dtype=np.uint64 )

if rank == 0:
	print('input data: %s' % argv[1])

	pd_in = pd.read_csv( argv[1], header=None ).to_numpy( dtype=np.float32, copy=True )

	# Force row_major
	pd_send_data = np.empty( pd_in.shape, dtype=np.float32 )
	pd_send_data[:,:] = pd_in[:,:]
	del pd_in

	np_send_data = np.loadtxt( argv[1], delimiter=',', dtype=np.float32 )

	#print('Node 0 read in numpy: \n', np_send_data.flags,'\n', np_send_data, '\n')
	print('Node 0 read in pandas: \n', pd_send_data.flags,'\n', pd_send_data, '\n')
	
	# Rank 0 calculates how much data is going to each node, and which array to sort by

	n_per_node = round( 2 * float(pd_send_data.shape[0]) / float(size) )
	sort_col = int( argv[2] )

	print( 'Size of input data: ', pd_send_data.shape[0] )
	print( 'N per node ', n_per_node )
	print( 'Sort by col: ', sort_col )

	size_col[0] = n_per_node
	size_col[1] = sort_col
	print( 'Sending to all: ',size_col)


# Rank 0 will do a "one to all" call saying how much data to expect, and which array to sort by
size_col = comm.bcast( size_col, root = 0 )
comm.Barrier()
time.sleep( rank*0.25 )
print( 'Rank %d received: %d - %d'%(rank, size_col[0], size_col[1]))
my_n = size_col[0]
sort_col = size_col[1]

# Create a recieving buffer with np.nan as placeholder
my_array = np.ones( ( my_n, 4 ), dtype=np.float32 ) * np.nan

# Rank 0 is scattering the initial data to all nodes
#send_data = np.transpose( send_data )
comm.Scatter( pd_send_data, my_array, root=0 )

comm.Barrier()
time.sleep( rank*0.25 )
print( 'Rank %d received my_array: \n'%rank, my_array )

# TODO: Each node will organize there array and determine which items are being sent where
def bin_my_array( my_array ):

	index_to_nodes = []
	for i in range( size ):
		index_to_nodes.append([])

	bins = np.linspace( 0, 1, size +1 )
	data_to_nodes = []

	for i in range( size ):
		data_to_nodes.append( my_array[(my_array[:,sort_col] < bins[i+1]) & (my_array[:,sort_col] >= bins[i])] )
	
	time.sleep( rank*0.25 )

	#'''
	# For initial printing and code development
	print( 'Rank %d array:'%rank)
	print( my_array )
	for i in range( size ):
		print( '%d -> %d' % ( rank, i ) )
		print( data_to_nodes[i] )
	#'''

	return data_to_nodes

data_to_nodes = bin_my_array( my_array )

# Clean up my array.
my_array *= np.nan


# Create integer array of how many items are going to each node
count_to_nodes = np.zeros( size, dtype=np.uint64 )
for i in range( size):
	count_to_nodes[i] = len( data_to_nodes[i] )

comm.Barrier()
time.sleep( rank*0.25 )
print( 'Rank %d count_to_nodes: ' % rank, count_to_nodes )

# Make all to all mpi call passing of number of items
# All nodes should know how many items they're getting
count_from_nodes = comm.alltoall( count_to_nodes )

# Test Print
comm.Barrier()
time.sleep( rank*0.25 )
print( 'Rank %d count_from_nodes: '%rank, count_from_nodes )

my_n = np.sum( count_from_nodes )
if my_n > my_array.shape[0]:
	print("Rank %d my_array not big enough!: %d"%(rank, my_n) )
	np.delete( my_array )
	my_array = np.ones( ( my_n, 4 ), dtype=np.float32 ) * np.nan

# TODO: Add items to your own large array again.

# Calc where my data should be saved
if rank == 0:
	my_i = 0
else:
	my_i = int( np.sum( count_from_nodes[0:rank] ) )


comm.Barrier()
time.sleep( rank*0.25 )
print( 'Rank %d saving own data to index: %d : %d' %( rank, int( my_i ), int( my_i + count_from_nodes[rank] ) ) )
my_array[ int(my_i) : int( my_i + count_from_nodes[rank] ), : ] = data_to_nodes[ rank ][:,:]
print( 'Rank %d saving own data: \n' % rank, my_array )
#my_array = data_to_nodes[ rank ]


comm.Barrier()
time.sleep( rank )

# TODO:  Loop through nodes, sending the subsection of array to node

# Just practice with rank 1 sending to rank 2 for now
for i in range( size ):
	#if rank == 1:
	# Skip self
	if i == rank:
		continue

	# Send the array, and move on
	comm.isend( data_to_nodes[i], dest=i, tag=25 )

	print( "Rank %d to %d: \n"%(rank, i),data_to_nodes[i] )

comm.Barrier()
time.sleep( rank )

# TODO:  Loop through nodes, gathering the subsections of arrays from each node and place in one large array

my_i = 0
for i in range( size ):

	# Skip self
	if rank == i:
		my_i += count_from_nodes[i]
		continue
	
	my_array[ int(my_i) : int(my_i + count_from_nodes[i]), : ] = comm.recv( source=i, tag=25 )
	my_i += count_from_nodes[i]

	print("Rank %d from %d:\n"%(rank,i), my_array )

my_array = my_array[ 0 : my_n , : ]

comm.Barrier()
time.sleep( rank*0.25 )
print( "Rank %d will sort: \n "%rank, my_array )

my_array = my_array[ np.argsort( my_array[:, sort_col] ) ]


comm.Barrier()
time.sleep( rank*0.25 )
print( "Rank %d sorted array!: \n "%rank, my_array )

