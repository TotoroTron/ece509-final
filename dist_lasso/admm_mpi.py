import numpy as np
import pandas as pd
import mpi4py.MPI as mpi

def z_minimization_step(x, y, rho, lambd):
    tau = lambd / rho
    vector = x + y / rho
    # Apply soft-thresholding element-wise
    return np.sign(vector) * np.maximum(np.abs(vector) - tau, 0)



def MPI_ADMM_Lasso(A, b, x, z, y, rho, lambd, epsilon, max_iters):
    Nb = A.shape[0] # A has shape (Nb, Nx)
    Nx = A.shape[1]
    list_primal_res = []
    list_dual_res = []
    list_x = []
    list_z = []

    iters = np.arange(0, max_iters) # max number of iterations
    # (A.T * A + rho * I)^(-1)
    cached_inv = np.linalg.inv(np.matmul(A.T, A) + rho*np.identity(Nx)) # O(n^3)

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
        if primal_res < epsilon and dual_res < epsilon:
            break
    list_x.append(x)
    return list_primal_res, list_dual_res, list_x, list_z



def ADMM_Lasso(A, b, x, z, y, rho, lambd, epsilon, max_iters):
    Nb = A.shape[0] # A has shape (Nb, Nx)
    Nx = A.shape[1]
    list_primal_res = []
    list_dual_res = []
    list_x = []
    list_z = []

    iters = np.arange(0, max_iters) # max number of iterations
    # (A.T * A + rho * I)^(-1)
    cached_inv = np.linalg.inv(np.matmul(A.T, A) + rho*np.identity(Nx)) # O(n^3)

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
        if primal_res < epsilon and dual_res < epsilon:
            break
    list_x.append(x)
    return list_primal_res, list_dual_res, list_x, list_z