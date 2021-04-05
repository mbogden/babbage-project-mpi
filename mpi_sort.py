# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
from sys import argv
import numpy as np
from mpi4py import MPI

# Global variable for now
n_total = 12  # Hard coded size of input file
n = 4 	# Hard coded size of everyone's beginning array
ci = 1  # Hard coded column number to sort by
 
# Grab useful things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print( 'I am rank %d of %d' % ( rank, size ) )
 
# Create data to broadcast
send_data = None
if rank == 0:
	print('input data: %s' % argv[1])
	send_data = np.loadtxt( argv[1], delimiter=',', dtype=np.float32 )
	print('Node 0 sending: \n',send_data,'\n')

recv_data = np.empty((3,4),dtype=np.float32)

comm.Scatter( send_data, recv_data, root=0 )

print(recv_data)
