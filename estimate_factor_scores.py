import argparse
import numpy as np
import os
import torch
from functools import partial
from skfda.representation.basis import Basis, BSplineBasis
from typing import Callable

from config import load_config
from utils import (
    gen_arrays,
    gen_tensors_as_arrays,
    get_generator,
)


def estimate_factors_pls(
        get_data_loader: Callable, 
        loads: np.ndarray,
        idx: np.ndarray
    ) -> np.ndarray:
    """Perform least squares regression at each time t:
            F_hat = (L @ Lt)_inv @ L @ X
    """
    temp = np.linalg.inv(loads[:,idx] @ loads[:,idx].T) @ loads
    facs = 0
    start = 0
    for data in get_data_loader():
        sz = len(data)
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        facs = facs + temp[:,idx_] @ data[idx_ - start]
        start += sz
    return facs


def estimate_factors_pgls(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        get_inv_cov_error_loader: Callable,
        idx: np.ndarray
    ) -> np.ndarray:
    """Perform generalized least squares regression at each time t:
            F_hat = (L @ B_inv @ Lt)_inv @ L @ B_inv @ X
    """

    # Get (L @ B_inv @ Lt)_inv
    temp = 0
    start = 0
    for rows in get_inv_err_cov_loader():
        sz = len(rows)
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        temp = temp + loads[:,idx] @ rows[np.ix_(idx_ - start, idx)].T @ loads[:,idx_].T
        start += sz
    temp = np.linalg.inv(temp)

    # Compute factors
    facs = 0
    start = 0
    for rows, data in zip(get_inv_cov_error_loader(), get_data_loader()):
        sz = len(rows)
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        facs = facs + temp @ loads[:,idx] @ rows[np.ix_(idx_ - start, idx)].T @ data[idx_ - start]
        start += sz
    return facs


def estimate_factors_rbels(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        idx: np.ndarray,
        basis: Basis, 
        gamma: float
    ) -> np.ndarray:
    """Perform F-on-S regression using a least squares criterion and a 
    roughness penalty:
            vec(A) = (H_2 x H_1 + D x I * gamma / K)_inv @ vec(H_3)
            F = A @ E
        where H_1 = L @ Lt
              H_2 = E @ Et
              H_3 = L @ X @ Et
              E is matrix containing discretized basis elements
              D is symmetric matrix of second derivative basis element IPs
    """
    n_basis = len(basis)
    n_facs = loads.shape[0]
    n_time = next(get_data_loader()).shape[1]

    # Build the following matrices:
    #   E --> E_{jt} = e_j(s_t)
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
    #   H1 = L @ Lt
    h1_mat = loads[:,idx] @ loads[:,idx].T

    # Compute basis coefficients for each sample
    temp = np.linalg.inv(
        np.kron(h2_mat, h1_mat) +
        np.kron(d_mat, gamma / n_facs * np.eye(n_facs))
    )
    h3_mat = 0
    start = 0
    for data in get_data_loader():
        sz = len(data)
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        h3_mat = h3_mat + loads[:,idx_] @ data[idx_ - start] @ e_mat.T 
        start += sz

    # Obtain factor coefficients, then return factors
    a_vec = temp @ h3_mat.flatten(order='F')
    a_mat = a_vec.reshape((n_facs, n_basis), order='F')
    return a_mat @ e_mat


def estimate_factors_rbegls(
        get_data_loader: Callable,
        loads: np.ndarray, 
        get_inv_err_cov_loader: Callable, 
        idx: np.ndarray,
        basis: Basis, 
        gamma: float
    ) -> np.ndarray:
    """Perform F-on-S regression using a generalized least squares criterion 
    and a roughness penalty:
            vec(A) = (H_2 x H_1 + D x I * gamma / K)_inv @ vec(H_3)
            F = A @ E
        where H_1 = L @ B_inv @ Lt
              H_2 = E @ Et
              H_3 = L @ B_inv @ X @ Et
              E is matrix containing discretized basis elements
              D is symmetric matrix of second derivative basis element IPs
    """
    n_basis = len(basis)
    n_facs = loads.shape[0]
    # n_space = loads.shape[1]
    n_space = len(idx)
    n_time = next(get_data_loader()).shape[1]

    # Build the following matrices:
    #   E --> E_{jt} = e_j(s_t)
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
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        h1_mat = h1_mat + loads[:,idx] @ rows[np.ix_(idx_ - start, idx)].T @ loads[:,idx_].T
        start += sz
    h1_mat = h1_mat / n_space ** 2

    # Compute basis coefficients for each sample
    temp = np.linalg.inv(
        np.kron(h2_mat, h1_mat) +
        np.kron(d_mat, gamma / n_facs * np.eye(n_facs))
    )
    h3_mat = 0
    start = 0
    for rows, data in zip(get_inv_err_cov_loader(), get_data_loader()):
        sz = len(rows)
        idx_mask = np.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        h3_mat = h3_mat + loads[:,idx] @ rows[np.ix_(idx_ - start, idx)].T @ data[idx_ - start] @ e_mat.T 
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
    parser.add_argument(
        '--est_methods', type=str, nargs='+',
        choices=['pls', 'pgls', 'rbels', 'rbegls'],
        default=['pls', 'pgls', 'rbels', 'rbegls']
    )
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--split', type=str)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Set directories/paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')

    # NOTE: (Regimes)
    #   (1) Estimate from (C_hat, L, B)
    #   (2) Estimate from (C_hat, L, B_hat)
    #   (3) Estimate from (C_hat, L_hat, B_hat)

    if args.regime == 1:
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path).numpy()
        get_inv_err_cov_loader = partial(
            get_generator,
            gen_fcn=gen_arrays,
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix='inv-err-cov_reg-1',
            batch_size=args.batch_size, 
            sort_by=('i', int)
        )

    if args.regime == 2:
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path).numpy()
        get_inv_err_cov_loader = partial(
            get_generator,
            gen_fcn=gen_arrays,
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix='inv-err-cov_reg-2',
            batch_size=args.batch_size, 
            sort_by=('i', int)
        )

    if args.regime == 3:
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-full.pth')
        loads = torch.load(path)['loads'].data.numpy().T
        get_inv_err_cov_loader = partial(
            get_generator,
            gen_fcn=gen_arrays,
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix='inv-err-cov_reg-3',
            batch_size=args.batch_size, 
            sort_by=('i', int)
        )

    # Create dataloaders for full data and split indices
    get_data_loader = partial(
        get_generator,
        gen_fcn=gen_tensors_as_arrays,
        dir=dir_data,
        prefix=f'data-space_split-full',
        batch_size=args.batch_size, 
        sort_by=('i', int)
    )
    if args.split in ['train', 'valid']:
        path = os.path.join(dir_data, f'idx-space_split-{args.split}_.pt')
        idx = torch.load(path).numpy()
    else: 
        idx = np.arange(loads.shape[1])

    # Generate saturated basis if using RBE methods
    n_time = next(get_data_loader()).shape[1]
    basis = BSplineBasis([0, 1], n_basis=n_time, order=4)

    if 'pls' in args.est_methods: 
        print("Estimating via PLS...")
        facs = estimate_factors_pls(get_data_loader, loads, idx)
        path = os.path.join(args.dir_out, f'facs_split-{args.split}_reg-{args.regime}_pls.pt')
        torch.save(torch.tensor(facs), path)

    if 'pgls' in args.est_methods: 
        print("Estimating via PGLS...")
        facs = estimate_factors_pgls(get_data_loader, loads, get_inv_err_cov_loader, idx)
        path = os.path.join(args.dir_out, f'facs_split-{args.split}_reg-{args.regime}_pgls.pt')
        torch.save(torch.tensor(facs), path)

    if 'rbels' in args.est_methods: 
        print("Estimating via RBELS...")
        facs = estimate_factors_rbels(get_data_loader, loads, idx, basis, args.gamma)
        path = os.path.join(args.dir_out, f'facs_split-{args.split}_reg-{args.regime}_rbels.pt')
        torch.save(torch.tensor(facs), path)

    if 'rbegls' in args.est_methods: 
        print("Estimating via RBEGLS...")
        facs = estimate_factors_rbegls(get_data_loader, loads, get_inv_err_cov_loader, idx, basis, args.gamma)
        path = os.path.join(args.dir_out, f'facs_split-{args.split}_reg-{args.regime}_rbegls.pt')
        torch.save(torch.tensor(facs), path)


    # TODO: Address erratic behavior near boundary for saturated basis when
    # gamma is zero. 
