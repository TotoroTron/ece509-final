import socket
import psutil
import platform
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpi4py import MPI
import admm_mpi as admm

np.random.seed(50) #for reproducibility

def print_cpu_info():
    process = psutil.Process()  # Get the current process
    cpu_num = process.cpu_num()  # Get the CPU number on which the current process is running
    return cpu_num

def plot_residuals(list_primal_res, list_dual_res, rho, lambd, max_iters):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(list_primal_res, label="Primal residual")
    ax.plot(list_dual_res, label="Dual residual")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Residual Error")
    ax.set_title(f"Convergence Plot (rho={rho}, lambda={lambd})")
    ax.legend()
    ax.grid(which='both', axis='both')  # Use logarithmic grid
    ax.set_ylim(10**-15, 10**3)  # Limit the y-axis to (10**-15, 10**3)
    ax.set_xlim(0, max_iters)
    return fig


def initialize_variables(Nx, Nb):
    """
    Initialize the variables for the ADMM algorithm.
    Input matrix A has shape (rows=Nb, cols=Nx).
    """

    A = np.random.randn(Nb, Nx)
    true_coeffs = np.random.randn(Nx)
    # A @ true_coeffs computes the dot product of each row of A with vector true_coeffs
    # which results in a new vector b of length Nb
    b = A @ true_coeffs + np.random.normal(0, 1, Nb) * 0.5
    x_init = np.zeros(Nx)
    z_init = np.zeros(Nx)

    # y represents the lagrange multiplier and dual variable
    y_init = np.zeros(Nx)
    
    return A, b, true_coeffs, x_init, z_init, y_init

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes
    hostname = socket.gethostname()  # Get the name of the node
    cpu_num = print_cpu_info()
    print(f"HELLO WORLD from process {rank} of {size} on {hostname}, running on CPU {cpu_num}, Processor: {platform.processor()}")

    M = 2000
    N = 4000
    Mi = M // size

    lambd = 0.1
    rho = 1.0

    if M % size != 0:
        if rank == 0:
            print("M is not evenly divisible by the number of processes.")
        exit()
    
    if rank == 0:
        print(f"Size of matrix A: (rows={M}, cols={N}), Size of submatrix: (rows={Mi}, cols={N})")

        A, b, true_coeffs, x_init, z_init, y_init = initialize_variables(N, M)

        # VALIDATION TEST SINGLE PROCESS
        start_time = time.time()
        list_primal_res, list_dual_res, list_x, list_z = admm.ADMM_Lasso(A, b, x_init, z_init, y_init, rho=rho, lambd=lambd, epsilon=1e-13, max_iters=3600)
        elapsed_time = time.time() - start_time
        print(f"Validation test elapsed time: {elapsed_time:.5f} seconds.")
        fig = plot_residuals(list_primal_res, list_dual_res, rho=rho, lambd=lambd, max_iters=3600)
        fig.savefig("validation_residuals.png", format="png", dpi=300, transparent=False)

    else:
        A = None
        b = None
        true_coeffs = None
        x_init = None
        z_init = None
        y_init = None
    
    ### ================= PARALLEL ZONE ================= ###

    local_A = np.empty((Mi, N), dtype='float64')

    # SCATTER THE ARRAY A
    start_time = time.time()
    comm.Scatter(A, local_A, root=0)
    elapsed_time = time.time() - start_time
    # Each process now has its own part of the array in recvbuf_A
    print(f"Time to scatter subblock A of shape {local_A.shape} to process {rank}: {elapsed_time:.5f} seconds")

    # SCATTER THE VECTOR b
    local_b = np.empty(Mi, dtype='float64')
    comm.Scatter(b, local_b, root=0)
    print(f"Process ", {rank}, " has received b of shape", local_b.shape)

    # BROADCAST THE VECTOR x_init
    local_x = comm.bcast(x_init, root=0)
    print(f"Process ", {rank}, " has received x of shape", local_x.shape)

    # BROADCAST THE VECTOR z_init
    local_z = comm.bcast(z_init, root=0)
    print(f"Process ", {rank}, " has received z of shape", local_z.shape)

    # BROADCAST THE VECTOR y_init
    local_y = comm.bcast(y_init, root=0)
    print(f"Process ", {rank}, " has received y of shape", local_y.shape)


    start_time = time.time()
    list_primal_res, list_dual_res, list_x, list_z = admm.MPI_ADMM_Lasso(comm, local_A, local_b, local_x, local_z, local_y, rho=rho, lambd=lambd, epsilon=1e-13, max_iters=3600)
    elapsed_time = time.time() - start_time
    print(f"MPI test elapsed time: {elapsed_time:.5f} seconds")
    fig = plot_residuals(list_primal_res, list_dual_res, rho=rho, lambd=lambd, max_iters=3600)
    fig.savefig(f"mpi_residuals_process_{rank}.png", format="png", dpi=300, transparent=False)




if __name__ == "__main__":
    main()
