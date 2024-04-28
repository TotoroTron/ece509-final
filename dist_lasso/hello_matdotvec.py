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

    if M % size != 0:
        if rank == 0:
            print("M is not evenly divisible by the number of processes.")
        exit()

    if rank == 0:
        # We want to calculate Ax-b
        A = np.random.normal(2000, 200, (M, N))
        x = np.random.normal(100, 10, N)
        # b = np.random.normal(500, 50, (N, 1))

        start_time = time.time()
        validation_dot = np.dot(A, x)
        elapsed_time = time.time() - start_time
        print(f"Validation dot: {validation_dot}")
        print(f"Validation shape: {validation_dot.shape}")
        print(f"Validation Elapsed time: {elapsed_time}")

        A_subblocks = np.array_split(A, 8, axis=0)
        # A_subblocks = [A[i * Mi: (i + 1) * Mi, :] for i in range(size)]
        
    else:
        A_subblocks = None
        x = None


    start_time = time.time()
    local_A_block = comm.scatter(A_subblocks, root=0)
    print(f"Process ", {rank}, " has received a subblock of shape", local_A_block.shape)
    elapsed_time = time.time() - start_time
    print(f"Time to scatter subblock A of shape {local_A_block.shape} to process {rank}: {elapsed_time:.5f} seconds")

    start_time = time.time()
    local_x = comm.bcast(x, root=0)
    print(f"Process ", {rank}, " has received x of shape", local_x.shape)
    elapsed_time = time.time() - start_time
    print(f"Time to broadcast vector x of shape {local_x.shape} to process {rank}: {elapsed_time:.5f} seconds")



    start_time = time.time()
    local_dot = np.dot(local_A_block, local_x)
    elapsed_time = time.time() - start_time
    print(f"Process {rank} has finished the dot product in {elapsed_time:.4f} seconds.")

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


