# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
from sys import argv
import numpy as np
from mpi4py import MPI

# Global variable for now
 
# Grab useful things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print( 'I am rank %d of %d' % ( rank, size ) )
 
# Create data to broadcast
send_data = None
size_col = np.zeros( 2, dtype=np.uint64 )

if rank == 0:
	print('input data: %s' % argv[1])
	send_data = np.loadtxt( argv[1], delimiter=',', dtype=np.float32 )
	print('Node 0 sending: \n',send_data,'\n')
	
	# Rank 0 calculates how much data is going to each node, and which array to sort by

	n_per_node = round( 2 * float(send_data.shape[0]) / float(size) )
	sort_col = int( argv[2] )

	print( 'Size of input data: ', send_data.shape[0] )
	print( 'N per node ', n_per_node )
	print( 'Sort by col: ', sort_col )

	size_col[0] = n_per_node
	size_col[1] = sort_col
	print( 'Sending ot all: ',size_col)


# Rank 0 will do a "one to all" call saying how much data to expect, and which array to sort by
size_col = comm.bcast( size_col, root = 0 )
print( 'Rank %d received: %d - %d'%(rank, size_col[0], size_col[1]))
my_n = size_col[0]
sort_col = size_col[1]

# Create a recieving buffer with np.nan as placeholder
my_array = np.ones( ( my_n, 4 ), dtype=np.float32 ) * np.nan

# Rank 0 is scattering the initial data to all nodes
comm.Scatter( send_data, my_array, root=0 )
print( my_array )

# TODO: Each node will organize there array and determine which items are being sent where
def get_count_to_nodes():

	index_to_nodes = []
	for i in range( size ):
		index_to_nodes.append([])

	bins = np.linspace( 0, 1, size +1 )

	# From Dr. Wallin
	'''
	i = 1
	a = np.random.rand(100,4)
	b = a[ ( a[:,i] < 0.4 ) & a[:,i] >= 0.2 ) ]
	'''

	# TODO:  Be sure to ignore NAN in array!
	for i in range( my_array.shape[0] ):
		val = my_array[i,sort_col]

		# TODO Calc which node this value should go to
		to_node = i % size

		index_to_nodes[to_node].append(i)

	return index_to_nodes

index_to_nodes = get_count_to_nodes()
count_to_nodes = np.zeros( size, dtype=np.uint64 )

if rank == 0:
	for i in range( size ):
		print(" Rank %d Going to node %d: "%(rank,i), index_to_nodes[i] )

# Create integer array of how many items are going to each node
for i in range( size):
	count_to_nodes[i] = len( index_to_nodes[i] )
print( count_to_nodes )

# Make all to all mpi call passing of number of items
# All nodes should know how many items they're getting
# TODO: Make sure this is sending the correct amount to each person
count_from_nodes = comm.alltoall( count_to_nodes )
print( count_from_nodes )

# Clean up my array.
my_array *= np.nan


# TODO:  Adjust my_array size to hold all the data
my_n = np.sum( count_from_nodes )
if my_n > my_array.shape[0]:
	print("Rank %d my_array not big enough!: %d"%(rank, my_n) )
# TODO: Add items to your own large array again.


# TODO:  Loop through nodes, sending the subsection of array to node

# Just practice with rank 1 sending to rank 2 for now
#for i in range( size ):
if rank == 1:
	# Skip self

	# TODO: Construct temporary array to send to everyone
	temp_array_1 = np.ones( ( count_to_nodes[1] , 4 )) * rank

	# Send the array, and move on
	comm.isend( temp_array_1, dest=2, tag=77 )

	print("Rank 1 sent to node 2: \n",temp_array_1)

# TODO: Create temporary array to receive.
temp_array_2 = np.ones( ( np.amax(count_from_nodes), 4 ), dtype=np.float32 ) * np.nan


# TODO:  Loop through nodes, gathering the subsections of arrays from each node and place in one large array

#for i in range( size ):
if rank == 2:

	# Skip self
	
	temp_array_2 = comm.recv( source=1, tag=77 )

	print("Rank 2 received from rank 1:\n", temp_array_2)

	# TODO: Add subarray to large array and move to next receive

	# my_array[ something : something+size of temp array ] = temp_array_2 

	# del temp_array


# TODO:  Sort large array.

