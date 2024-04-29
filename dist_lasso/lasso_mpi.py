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


def plot_residuals(node, list_primal_res, list_dual_res, p_star, p_stop, stop_iter, rho, lambd, epsilon, max_iters):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(list_primal_res, label="Primal residual")
    ax.plot(list_dual_res, label="Dual residual")
    
    # Plot a vertical line at stop_iter
    ax.axvline(x=stop_iter, color='red', linestyle='--', label=f"Stopping criterion met at k = {stop_iter}")

    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Residual Error")
    ax.set_title(f"Convergence Plot for Node {node} \n(rho={rho}, lambda={lambd}, epsilon={epsilon}), p_star = {p_star:.2e}, p_stop = {p_stop:.2e}")
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

def objective_function(A, b, x, lambd):
    print("Shape of A:", A.shape, "Shape of b:", b.shape)
    # print("Shape of x:", x.shape)
    # lasso function: 0.5 * ||A x - b||^2 + lambd * ||x||_1
    return 0.5 * np.sum((np.matmul(A, x) - b)**2) + lambd * np.sum(np.abs(x))

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes
    hostname = socket.gethostname()  # Get the name of the node
    cpu_num = print_cpu_info()
    print(f"HELLO WORLD from process {rank} of {size} on {hostname}, running on CPU {cpu_num}, Processor: {platform.processor()}")

    M = 4000
    N = 2000
    Mi = M // size

    lambd = 0.1
    rho = 1.0
    epsilon=1e-2 # 0.001 acceptable residual for convergence
    max_iters = 2000

    if M % size != 0:
        if rank == 0:
            print("M is not evenly divisible by the number of processes.")
        exit()
    
    if rank == 0:
        print(f"Size of matrix A: (rows={M}, cols={N}), Size of submatrix: (rows={Mi}, cols={N})")

        A, b, true_coeffs, x_init, z_init, y_init = initialize_variables(N, M)

        # VALIDATION TEST SINGLE PROCESS
        start_time = time.time()
        list_primal_res, list_dual_res, list_x, x_star, x_stop, stop_iter = admm.ADMM_Lasso(A, b, x_init, z_init, y_init, rho=rho, lambd=lambd, epsilon=epsilon, max_iters=max_iters)
        elapsed_time = time.time() - start_time
        print(f"Validation test elapsed time: {elapsed_time:.5f} seconds.")
        p_stop = objective_function(A, b, x_stop, lambd)
        p_star = objective_function(A, b, x_star, lambd)
        fig = plot_residuals(rank, list_primal_res, list_dual_res, p_star, p_stop, stop_iter, rho=rho, lambd=lambd, epsilon=epsilon, max_iters=max_iters)
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

    # PERFORM THE MPI TEST
    start_time = time.time()
    list_primal_res, list_dual_res, list_x, x_star, x_stop, stop_iter = admm.MPI_ADMM_Lasso(comm, local_A, local_b, local_x, local_z, local_y, rho=rho, lambd=lambd, epsilon=epsilon, max_iters=max_iters)
    elapsed_time = time.time() - start_time
    print(f"MPI test elapsed time for process {rank}: {elapsed_time:.5f} seconds")


    # GATHER THE RESULTS TO ROOT FOR PLOTTING
    if rank == 0:
        all_primal_res = []
        all_dual_res = []
        all_list_x = []
        all_x_stars = []
        all_x_stops = []
        all_stop_iters = []
    else:
        all_primal_res = None
        all_dual_res = None
        all_list_x = None
        all_x_stars = None
        all_x_stops = None
        all_stop_iters = None

    all_primal_res = comm.gather(list_primal_res, root=0)
    all_dual_res = comm.gather(list_dual_res, root=0)
    all_list_x = comm.gather(list_x, root=0)
    all_x_stars = comm.gather(x_star, root=0)
    all_x_stops = comm.gather(x_stop, root=0)
    all_stop_iters = comm.gather(stop_iter, root=0)

    if rank == 0:
        # all_p_stars = objective_function(A, b, all_x_stars, lambd)
        for i in range(size):
            # ith node/process
            print(len(all_x_stars[i]))
            print((all_x_stops[i]))
            p_star = objective_function(A, b, all_x_stars[i], lambd)
            p_stop = objective_function(A, b, all_x_stops[i], lambd)
            fig = plot_residuals(i, all_primal_res[i], all_dual_res[i], p_star, p_stop, all_stop_iters[i], rho=rho, lambd=lambd, epsilon=epsilon, max_iters=max_iters)
            fig.savefig(f"mpi_residuals_process_{i}.png", format="png", dpi=300, transparent=False)
        
        for i in range(size):
            # plot objective function over x
            list_objfunc_eval = [objective_function(A, b, x, lambd) for x in all_list_x[i]]



if __name__ == "__main__":
    main()
