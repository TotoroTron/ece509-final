import numpy as np
import pandas as pd
import mpi4py.MPI as mpi
import time

def z_minimization_step(x, y, rho, lambd):
    tau = lambd / rho
    vector = x + y / rho
    # Apply soft-thresholding element-wise
    return np.sign(vector) * np.maximum(np.abs(vector) - tau, 0)



def MPI_ADMM_Lasso(comm, A, b, x_init, z_init, y_init, rho, lambd, epsilon, max_iters):
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Total number of processes

    list_primal_res = []
    list_dual_res = []
    list_x = []
    list_z = []

    x = x_init
    z = z_init
    y = y_init

    x_star = None
    conv_iter = None
    convergence_flag = False


    iters = np.arange(0, max_iters) # max number of iterations
    # (A.T * A + rho * I)^(-1)
    inv_start_time = time.time()
    cached_inv = np.linalg.inv(np.matmul(A.T, A) + rho*np.identity(A.shape[1]))
    inv_elapsed_time = time.time() - inv_start_time
    print(f"Time to invert matrix (A.T * A + rho * I): {inv_elapsed_time:.5f} seconds")

    for k in iters:
        # local variable x update
        x = np.matmul(cached_inv, np.matmul(A.T, b) + rho*z - y) # x-minimization step
        z_prev = z.copy() # Store the previous value of z for convergence checking

        # gather local x and local y from all processes
        x_all = comm.gather(x, root=0) # a list of vectors
        y_all = comm.gather(y, root=0)
        
        if rank == 0:
            # average x and y across all processes
            x_avg = np.mean(x_all, axis=0)
            y_avg = np.mean(y_all, axis=0)
            z = z_minimization_step(x_avg, y_avg, rho, lambd) # z-minimization step
        else:
            z = None

        # broadcast z back to all processes
        z = comm.bcast(z, root=0)

        # local dual variable update
        y = y + rho * (x - z)

        # calculate local residuals
        primal_res = np.linalg.norm(x - z, ord=2)
        dual_res = rho * np.linalg.norm(z - z_prev, ord=2)

        list_primal_res.append(primal_res)
        list_dual_res.append(dual_res)

        # Check for convergence
        if primal_res < epsilon and dual_res < epsilon and not convergence_flag:
            x_star = x
            conv_iter = k
            convergence_flag = True

    return list_primal_res, list_dual_res, x_star, conv_iter



def ADMM_Lasso(A, b, x, z, y, rho, lambd, epsilon, max_iters):
    Nb = A.shape[0] # A has shape (Nb, Nx)
    Nx = A.shape[1]
    list_primal_res = []
    list_dual_res = []
    list_x = []
    list_z = []

    x_star = None
    conv_iter = None
    convergence_flag = False

    iters = np.arange(0, max_iters) # max number of iterations
    # (A.T * A + rho * I)^(-1)
    start_time = time.time()
    cached_inv = np.linalg.inv(np.matmul(A.T, A) + rho*np.identity(Nx)) # ~O(n^2+)
    elapsed_time = time.time() - start_time
    print(f"Time to invert matrix: {elapsed_time:.5f} seconds")

    for k in iters:
        # primal variable updates
        x = np.matmul(cached_inv, np.matmul(A.T, b) + rho*z - y) # x-minimization step
        z_prev = z.copy() # Store the previous value of z for convergence checking
        z = z_minimization_step(x, y, rho, lambd) # z-minimization step

        # dual variable update
        y = y + rho * (x - z)

        # calculate residuals
        primal_res = np.linalg.norm(x - z, ord=2)
        dual_res = rho * np.linalg.norm(z - z_prev, ord=2)

        list_primal_res.append(primal_res)
        list_dual_res.append(dual_res)

        # Check for convergence
        if primal_res < epsilon and dual_res < epsilon and not convergence_flag:
            x_star = x
            conv_iter = k
            convergence_flag = True
        
    return list_primal_res, list_dual_res, x_star, conv_iter