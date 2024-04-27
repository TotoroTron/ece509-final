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
    N = 1600
    M = N // size

    if rank == 0:
        A = np.random.normal(500, 100, (N, N))
        x = np.random.randn(N)
        b = np.random.randn(N)

        validation_mean = np.mean(A)
        print(f"Validation mean: {validation_mean}")
        print(f"Validation shape: {validation_mean.shape}")

        subblocks = []
        for i in range(size):
            # SPLITTING ACROSS EXAMPLES
            """
            # Numpy slicing syntax: A[rows=0:400, cols=:]
            # subblock[0] = A[0:400, :]
            # subblock[1] = A[400:800, :], etc.
            """
            subblocks.append(A[i * M: (i + 1) * M, :])
    else:
        subblocks = None
        x = None
        b = None

    local_block = comm.scatter(subblocks, root=0)
    print(f"Process ", {rank}, " has received a subblock of shape", local_block.shape, "\n")

    local_mean = np.mean(local_block)
    print(f"Process ", {rank}, " has a local mean of ", local_mean, "\n")

    local_means = comm.gather(local_mean, root=0)

    if rank == 0:
        print("Length of local means: ", len(local_means))
        global_mean = np.mean(local_means)

        error = global_mean - validation_mean
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
