# Code taken and modified from:
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
 
import numpy as np
from mpi4py import MPI
 
# Grab useful things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print( 'I am rank %d of %d' % ( rank, size ) )
 
# Create data to broadcast
if rank == 0:
    data = np.arange( size, dtype='i' )
 
# Create empty array to receive
else:
    data = np.empty( size, dtype='i' )
 
# Broadcast data
comm.Bcast( data, root=0 )
 
# Check if data was sent correctly
print("My rank should match value: %d - %d" % ( rank, data[ rank ] ) )