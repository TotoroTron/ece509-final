from mpi4py import MPI
import socket
import psutil
import platform
import numpy as np
import time

def print_cpu_info():
    process = psutil.Process()  # Get the current process
    cpu_num = process.cpu_num()  # Get the CPU number on which the current process is running
    return cpu_num

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes
    hostname = socket.gethostname()  # Get the name of the node
    cpu_num = print_cpu_info()
    print(f"HELLO WORLD from process {rank} of {size} on {hostname}, running on CPU {cpu_num}, Processor: {platform.processor()}")

    # assert size == 8
    M = 14000
    N = 10000
    Mi = M // size

    # Check if the number of nodes is appropriate
    assert M % size == 0, "The matrix cannot be evenly split across the provided number of processes."

    if rank == 0:
        sendbuf_A = np.random.normal(2000, 200, (M, N))
        x = np.random.normal(100, 10, N)
 
        start_time = time.time()
        validation_dot = np.dot(sendbuf_A, x)
        elapsed_time = time.time() - start_time
        print(f"Validation dot: {validation_dot}")
        print(f"Validation shape: {validation_dot.shape}")
        print(f"Validation Elapsed time: {elapsed_time}")
    else:
        sendbuf_A = None
        x = np.empty(N, dtype='float64')

    recvbuf_A = np.empty((Mi, N), dtype='float64')

    # SCATTER THE ARRAY A
    start_time = time.time()
    comm.Scatter(sendbuf_A, recvbuf_A, root=0)
    elapsed_time = time.time() - start_time
    # Each process now has its own part of the array in recvbuf
    print(f"Time to scatter subblock A of shape {recvbuf_A.shape} to process {rank}: {elapsed_time:.5f} seconds")

    # BROADCAST THE VECTOR x
    start_time = time.time()
    local_x = comm.bcast(x, root=0)
    print(f"Process ", {rank}, " has received x of shape", local_x.shape)
    elapsed_time = time.time() - start_time
    print(f"Time to broadcast vector x of shape {local_x.shape} to process {rank}: {elapsed_time:.5f} seconds")

    # COMPUTE THE LOCAL DOTS
    start_time = time.time()
    local_dot = np.dot(recvbuf_A, local_x)
    elapsed_time = time.time() - start_time
    print(f"Process {rank} has finished the dot product in {elapsed_time:.4f} seconds.")

    # GATHER THE LOCAL DOTS
    start_time = time.time()
    local_dots = comm.gather(local_dot, root=0)
    elapsed_time = time.time() - start_time
    print(f"Time to gather: {elapsed_time:.4f} seconds")

    if rank == 0:
        print("Length of local dots: ", len(local_dots))
        global_dot = np.concatenate(local_dots, axis=0)

        error_norm = np.linalg.norm(global_dot - validation_dot)
        print(f"Error Norm: {error_norm}")
    
if __name__ == "__main__":
    main()
