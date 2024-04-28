import socket
import psutil
import platform

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

def plot_residuals(A_shape, list_primal_res, list_dual_res, rho, lambd, max_iters):
    plt.figure(figsize=(8,8))
    plt.plot(list_primal_res, label="Primal residual")
    plt.plot(list_dual_res, label="Dual residual")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Residual Error")
    plt.title(f"Convergence Plot (Shape A: (rows={A_shape[0]}, cols={A_shape[1]})(rho={rho}, lambda={lambd})")
    plt.legend()
    plt.grid(which='both', axis='both')  # Enable grid for log scale
    plt.ylim(10**-15, 10**3)  # Limit the y-axis to this range.
    plt.xlim(0, max_iters)
    plt.savefig('residuals.png', format='png', dpi=300, transparent=True, quality=95)


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

    # assert size == 8
    M = 14000
    N = 10000
    Mi = M // size

    if M % size != 0:
        if rank == 0:
            print("M is not evenly divisible by the number of processes.")
        exit()
    
    if rank == 0:
        A, b, true_coeffs, x_init, z_init, y_init = initialize_variables(N, M)
        lambd = 0.1
        rho = 1.0
        list_primal_res, list_dual_res, list_x, list_z = admm.ADMM_Lasso(x_init, z_init, y_init, rho=rho, lambd=lambd, epsilon=1e-13, max_iters=3600)


if __name__ == "__main__":
    main()
