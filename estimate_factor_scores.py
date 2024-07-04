import argparse
import numpy as np
import os
import torch
from functools import partial
from skfda.representation import FDataBasis, FDataGrid
from skfda.representation.basis import Basis, BSplineBasis
from typing import Callable, Generator

from config import load_config
from utils.utils import (
    get_array_gen,
    get_tensor_as_array_gen
)


def estimate_factors_pls(
        get_data_loader: Callable, 
        loads: np.ndarray
    ) -> np.ndarray:
    """Perform least squares regression at each time t:
            F_hat = (L @ Lt)_inv @ L @ X
    """
    temp = np.linalg.inv(loads @ loads.T) @ loads
    facs = 0
    start = 0
    for data in get_data_loader():
        sz = len(data)
        facs = facs + temp[:,start:(start + len(data))] @ data
        start += sz
    return facs


def estimate_factors_pgls(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        get_inv_cov_error_loader: Callable
    ) -> np.ndarray:
    """Perform generalized least squares regression at each time t:
            F_hat = (L @ B_inv @ Lt)_inv @ L @ B_inv @ X
    """

    # Get (L @ B_inv @ Lt)_inv
    temp = 0
    start = 0
    for rows in get_inv_err_cov_loader():
        sz = len(rows)
        temp = temp + loads @ rows.T @ loads.T[start:(start + sz)]
        start += sz
    temp = np.linalg.inv(temp)

    # Compute factors
    facs = 0
    for rows, data in zip(get_inv_cov_error_loader(), get_data_loader()):
        facs = facs + temp @ loads @ rows.T @ data
    return facs


# def estimate_factors_rbels_old(
#         get_data_loader: Callable, 
#         loads: np.ndarray, 
#         basis: Basis, 
#         alpha: float
#     ) -> np.ndarray:
#     """Perform F-on-S regression using a least squares criterion and a 
#     roughness penalty:
#             vec(A) = (B x LLt + D x I * alpha / K)_inv @ vec(L @ X @ B)
#             F = A @ E
#         where E is matrix containing discretized basis elements
#               B is symmetric matrix of basis element IPs
#               D is symmetric matrix of second derivative basis element IPs
#               X is matrix of basis coefficients for data
#               x denotes outer product
#     """

#     n_facs = loads.shape[0]
#     n_basis = len(basis)

#     # Construct two matrices: 
#     #   B where [B]_{ij} = <e_i, e_j>
#     #   D where [D]_{ij} = <e_i^{''}, e_j^{''}>
#     grid = np.linspace(0, 1, 1001)  # sufficiently dense grid
#     vals_b = basis(grid)[:,:,0]
#     vals_d = basis.derivative(order=2)(grid)[:,:,0]
#     b_mat = np.zeros((n_basis, n_basis))
#     d_mat = np.zeros((n_basis, n_basis))
#     for i in range(n_basis):
#         for j in range(n_basis):
#             b_mat[i,j] = np.mean(vals_b[i] * vals_b[j])
#             d_mat[i,j] = np.mean(vals_d[i] * vals_d[j])

#     # Compute (B x LLt + D x I * alpha / K)_inv
#     temp1 = np.linalg.inv( 
#         np.kron(b_mat, loads @ loads.T) + 
#         np.kron(d_mat, alpha / n_facs * np.eye(n_facs))
#     )

#     # print(f"d_mat.shape = {d_mat.shape}")
#     # print(f"temp1.shape = {temp1.shape}")
#     # return

#     # Compute factors in batches
#     a_vec = 0
#     start = 0
#     for b, data in enumerate(get_data_loader()): 

#         if b == 0:
#             grid_time = np.linspace(0, 1, data.shape[1])

#         # Get sz rows of x_mat
#         fdata = FDataGrid(data, grid_time)
#         sz = len(data)
#         x_mat = np.zeros((sz, n_basis))
#         for m in range(sz):  # TODO: Vectorize this
#             x_mat[m] = fdata[m].to_basis(basis).coefficients.squeeze()

#         # Compute sum-wise batch of L @ X @ B
#         temp2 = (loads[:, start:(start + sz)] @ x_mat @ b_mat).flatten(order='F')

#         # Add sum-wise batch to a_vec
#         a_vec = a_vec + temp1 @ temp2

#         start += sz
    
#     a_mat = a_vec.reshape((n_facs, n_basis), order='F')

#     # Discretize basis representation of factors
#     fdbasis = FDataBasis(basis, coefficients=a_mat)
#     return fdbasis(grid_time)[:,:,0]


def estimate_factors_rbels(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        basis: Basis, 
        alpha: float
    ) -> np.ndarray:
    """Perform F-on-S regression using a least squares criterion and a 
    roughness penalty:
            vec(A) = (B x LLt + D x I * alpha / K)_inv @ vec(L @ X @ B)
            F = A @ E
        where E is matrix containing discretized basis elements
              B is symmetric matrix of basis element IPs
              D is symmetric matrix of second derivative basis element IPs
              X is matrix of basis coefficients for data
              x denotes outer product
    """
    n_basis = len(basis)
    n_facs = loads.shape[0]
    n_time = next(get_data_loader()).shape[1]

    # Build the following matrices:
    # #   E --> E_{jt} = e_j(s_t)
    grid_time = np.linspace(0, 1, n_time)
    e_mat = basis(grid_time)[:,:,0]
    #   D --> D_{ij} = <e_i^{''}, e_j^{''}>
    grid = np.linspace(0, 1, 1001)  # sufficiently dense grid
    vals_d = basis.derivative(order=2)(grid)[:,:,0]
    d_mat = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            d_mat[i,j] = np.mean(vals_d[i] * vals_d[j])
    #   H2 = E @ Et
    h2_mat = e_mat @ e_mat.T    
    #   H1 = L @ B_inv Lt
    h1_mat = loads @ loads.T

    # Compute basis coefficients for each sample
    temp = np.linalg.inv(
        np.kron(h2_mat, h1_mat) +
        np.kron(d_mat, alpha / n_facs * np.eye(n_facs))
    )
    h3_mat = 0
    start = 0
    for data in get_data_loader():
        sz = len(data)
        h3_mat = h3_mat + loads[:,start:(start + sz)] @ data @ e_mat.T 
        start += sz

    # Obtain factor coefficients, then return factors
    a_vec = temp @ h3_mat.flatten(order='F')
    a_mat = a_vec.reshape((n_facs, n_basis), order='F')
    return a_mat @ e_mat



def estimate_factors_rbegls(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        basis: Basis, 
        get_inv_err_cov_loader: Callable, 
        alpha: float
    ) -> np.ndarray:
    """Perform F-on-S regression using a generalized least squares criterion 
    and a roughness penalty:
            vec(A) = (H_2 x H_1 + D x I * alpha / K)_inv @ vec(H_3)
        where H_1 = L @ B_inv Lt
              H_2 = E @ Et
              H_3 = L @ B_inv @ X @ Et
              E is matrix containing discretized basis elements
              D is symmetric matrix of second derivative basis element IPs

    """
    n_basis = len(basis)
    n_facs = loads.shape[0]
    n_space = loads.shape[1]
    n_time = next(get_data_loader()).shape[1]

    # Build the following matrices:
    # #   E --> E_{jt} = e_j(s_t)
    grid_time = np.linspace(0, 1, n_time)
    e_mat = basis(grid_time)[:,:,0]
    #   D --> D_{ij} = <e_i^{''}, e_j^{''}>
    grid = np.linspace(0, 1, 1001)  # sufficiently dense grid
    vals_d = basis.derivative(order=2)(grid)[:,:,0]
    d_mat = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            d_mat[i,j] = np.mean(vals_d[i] * vals_d[j])
    #   H2 = E @ Et
    h2_mat = e_mat @ e_mat.T    
    #   H1 = L @ B_inv @ Lt
    h1_mat = 0
    start = 0
    for rows in get_inv_err_cov_loader():
        sz = len(rows)
        h1_mat = h1_mat + loads @ rows.T @ loads.T[start:(start + sz)]
        start += sz
    h1_mat = h1_mat / n_space ** 2

    # Compute basis coefficients for each sample
    temp = np.linalg.inv(
        np.kron(h2_mat, h1_mat) +
        np.kron(d_mat, alpha / n_facs * np.eye(n_facs))
    )
    h3_mat = 0
    start = 0
    for rows, data in zip(get_inv_err_cov_loader(), get_data_loader()):
        sz = len(rows)
        h3_mat = h3_mat + loads @ rows.T @ data @ e_mat.T 
        start += sz
    h3_mat = h3_mat / n_space ** 2

    # Obtain factor coefficients, then return factors
    a_vec = temp @ h3_mat.flatten(order='F')
    a_mat = a_vec.reshape((n_facs, n_basis), order='F')
    return a_mat @ e_mat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--dir_truth', type=str)
    parser.add_argument(
        '--est_methods', type=str, nargs='+',
        choices=['pls', 'pgls', 'rbels', 'rbegls'],
        default=['pls', 'pgls', 'rbels', 'rbegls']
    )
    parser.add_argument('--n_time', type=int)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--split', type=str)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Validate command-line arguments
    #   - est_method compatible with alpha
    #   - "true" flags compatible with --dir_truth

    # Set directories/paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')

    # NOTE: (Regimes)
    #   (1) Estimate from (C_hat, L, B)
    #   (2) Estimate from (C_hat, L, B_hat)
    #   (3) Estimate from (C_hat, L_hat, B_hat)

    # Create data loader
    get_data_loader = partial(
        get_tensor_as_array_gen,
        dir=dir_data, 
        prefix=f'data-space-{args.split}', 
        batch_size=args.batch_size
    )

    if args.regime == 1:
        path = os.path.join(args.dir_truth, 'loads.pt')
        loads = loads = torch.load(path).numpy()
        get_inv_err_cov_loader = partial(
            get_array_gen, 
            dir=args.dir_truth,
            prefix='inv-err-cov_r1',
            batch_size=args.batch_size
        )

    if args.regime == 2:
        path = os.path.join(args.dir_truth, 'loads.pt')
        loads = torch.load(path).numpy().T
        get_inv_err_cov_loader = partial(
            get_array_gen, 
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix='inv-err-cov_r2',
            batch_size=args.batch_size
        )

    if args.regime == 3:
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-{args.split}.pth')
        loads = torch.load(path)['loads'].data.numpy()
        get_inv_err_cov_loader = partial(
            get_array_gen, 
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix='inv-err-cov_r3',
            batch_size=args.batch_size
        )

    # Generate saturated basis if using RBE methods
    # TODO: Basis can be saturated for RBEGLS, but not for RBELS. 
    # Revisit when 
    basis = BSplineBasis([0, 1], n_basis=args.n_time, order=4)

    if 'pls' in args.est_methods: 
        print("Estimating via PLS...")
        facs = estimate_factors_pls(get_data_loader, loads)
        path = os.path.join(args.dir_out, f'facs_{args.split}_r{args.regime}_pls.pt')
        torch.save(torch.tensor(facs), path)

    if 'pgls' in args.est_methods: 
        print("Estimating via PGLS...")
        facs = estimate_factors_pgls(get_data_loader, loads, get_inv_err_cov_loader)
        path = os.path.join(args.dir_out, f'facs_{args.split}_r{args.regime}_pgls.pt')
        torch.save(torch.tensor(facs), path)

    if 'rbels' in args.est_methods: 
        print("Estimating via RBELS...")
        facs = estimate_factors_rbels(get_data_loader, loads, basis, args.alpha)
        path = os.path.join(args.dir_out, f'facs_{args.split}_r{args.regime}_rbels.pt')
        torch.save(torch.tensor(facs), path)

    if 'rbegls' in args.est_methods: 
        print("Estimating via RBEGLS...")
        facs = estimate_factors_rbegls(get_data_loader, loads, basis, get_inv_err_cov_loader, args.alpha)
        path = os.path.join(args.dir_out, f'facs_{args.split}_r{args.regime}_rbegls.pt')
        torch.save(torch.tensor(facs), path)


