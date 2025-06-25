import argparse
import numpy as np
import os
import torch
from functools import partial
from skfda.representation.basis import Basis, BSplineBasis
from typing import Callable

from config import load_config
from utils import (
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


def estimate_factors_rbels(
        get_data_loader: Callable, 
        loads: np.ndarray, 
        idx: np.ndarray,
        basis: Basis, 
        gammas: np.ndarray
    ) -> np.ndarray:
    """Perform F-on-S regression using a least squares criterion and a 
    roughness penalty:
            vec(A) = (H_2 x H_1 + D x I * gammas / K)_inv @ vec(H_3)
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
        np.kron(d_mat, gammas / n_facs * np.eye(n_facs))
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



def get_fac_fname(split, method, rot_method, regime, fold=None, sub_num=None):
    fname = f'facs_split-{split}'
    if split == 'train':
        fname += f'_v-{fold}'
    fname += f'_m-{method}_rot-{rot_method}_reg-{regime}'
    if sub_num is not None: 
        fname += f'_n-{sub_num}'
    return fname + '.pt' 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--est_method', type=str, choices=['pls', 'rbels'])
    parser.add_argument('--sub_nums', type=int, nargs='+')
    # parser.add_argument('--gammas', type=float, nargs='+')
    parser.add_argument('--path_gammas', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--rot_method', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--agg_subs', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set directories/paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')

    # Set gammas
    if 'rbels' in args.est_method: 
        if args.path_gammas: 
            gammas = torch.load(args.path_gammas).numpy()
        else: 
            raise Exception("Must pass gamma path!")
        if gammas.shape[0] != len(args.sub_nums):
            raise Exception(f"Ivalid gamma shape: {gammas.shape}")

    # NOTE: (Regimes)
    #   (1) Estimate from (C_hat, L, B)
    #   (2) Estimate from (C_hat, L, B_hat)
    #   (3) Estimate from (C_hat, L_hat, B_hat)

    # TODO: Remove regime 2 and rename regime 3 to regime 2

    if args.regime == 1:
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path).numpy()
        if args.path_mask is not None: 
            mask = torch.flatten(torch.load(args.path_mask)).numpy()
            loads = loads[:,mask]

    if args.regime == 2:
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path).numpy()
        if args.path_mask is not None: 
            mask = torch.flatten(torch.load(args.path_mask)).numpy()
            loads = loads[:,mask]

    if args.regime == 3:
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-full-{args.rot_method}-smooth-shrink.pth')
        loads = torch.load(path)['loads'].numpy().T


    # Get spatial indices
    if args.split in ['train', 'valid']:
        path = os.path.join(dir_data, f'idx-space_split-{args.split}_v-{args.fold}_.pt')
        idx = torch.load(path).numpy()
    else: 
        idx = np.arange(loads.shape[1])

    for i, n in enumerate(args.sub_nums):

        # Create dataloaders
        get_data_loader = partial(
            get_generator,
            gen_fcn=gen_tensors_as_arrays,
            dir=dir_data,
            prefix=f'data-space_split-full_n-{n}_',
            batch_size=args.batch_size, 
            sort_by=('i', int)
        )

        # Estimate factor scores
        if args.est_method == 'pls': 
            facs = estimate_factors_pls(get_data_loader, loads, idx)
        if args.est_method == 'rbels':
            n_time = next(get_data_loader()).shape[1]
            basis = BSplineBasis([0, 1], n_basis=int(0.95*n_time), order=4)
            facs = estimate_factors_rbels(get_data_loader, loads, idx, basis, gammas[i])
        
        # Save factors
        fname = get_fac_fname(args.split, args.est_method, args.rot_method, args.regime, args.fold, n)
        path = os.path.join(args.dir_out, 'est-facs', fname)
        torch.save(torch.tensor(facs).to(torch.float32), path)


    if args.agg_subs: 

        facs = []
        for n in args.sub_nums: 

            # Read subject factor file then remove
            fname = get_fac_fname(args.split, args.est_method, args.rot_method, args.regime, args.fold, n)
            path = os.path.join(args.dir_out, 'est-facs', fname)
            facs.append(torch.load(path))
            os.remove(path)

        # Aggregate factors
        facs = torch.stack(facs, dim=0)
        fname = get_fac_fname(args.split, args.est_method, args.rot_method, args.regime, args.fold)
        path = os.path.join(args.dir_out, fname)
        torch.save(facs, path)
