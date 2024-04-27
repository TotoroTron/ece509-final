from mpi4py import MPI
import socket
import psutil
import platform
import numpy as np

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

    assert size == 8
    N = 4000
    M = N // size

    if rank == 0:
        A = np.random.randn(N, N)
        x = np.random.randn(N)
        b = np.random.randn(N)

        subAs = [A[:, i * M:(i + 1) * M] for i in range(8)]
        subxs = [x[i * M:(i + 1) * M] for i in range(8)]
    else:
        subA = np.empty((N,M))
        subx = np.empty(M)
        subAs = None
        subxs = None
        A = None
        x = None
    
    # Scatter the subblocks and subvectors to all processes
    comm.Scatter(subAs, subA, root=0)
    comm.Scatter(subxs, subx, root=0)

    # Compute the local product
    local_result = np.dot(subA, subx)

    # Gather the results
    results = None
    if rank == 0:
        results = np.empty((size, M))
    comm.Gather(local_result, results, root=0)

    # Final summation at the root process
    if rank == 0:
        final_result = np.sum(results.reshape(size, M), axis=0)
        print("MPI Final Result:", final_result)

if __name__ == "__main__":
    main()
