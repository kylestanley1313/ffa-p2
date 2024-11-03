import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch
from functools import partial

from config import load_config
from utils import (
    execute_script, 
    get_generator,
    gen_arrays,
    gen_tensors_as_arrays,
)


GAMMAS = [
    1e-12,
    1e-11,
    1e-10, 
    1e-9, 
    1e-8, 
    1e-7, 
    1e-6, 
    1e-5, 
    1e-4,
]


def compute_ls_loss(idx, get_data_loader, loads, facs, skip_batching=False):
    """Computes least squares loss:
            1/(MT) trace((X - LF)^T (X - LF))
    """
    n_idx = len(idx)
    n_time = facs.shape[1]

    if skip_batching: 

        # Aggregate data
        data_list = []
        for data in get_data_loader():
            data_list.append(data)
        data = np.concatenate(data_list)

        # Compute loss
        errs = data[idx] - loads.T[idx] @ facs
        out = np.sum(errs ** 2)

    else: 
        out = 0
        start = 0
        for data in get_data_loader():
            sz = len(data)
            idx_mask = np.logical_and(idx >= start, idx < start + sz)
            idx_ = idx[idx_mask]
            errs = data[idx_ - start] - loads.T[idx_] @ facs
            out += np.sum(errs ** 2)
            start += sz
    
    return out / n_idx / n_time


def compute_gls_loss(idx, get_data_loader, get_inv_err_cov_loader, loads, facs, skip_batching=False):
    """Computes generalized least squares loss:
            1/(MT) trace((X - LF)^T B_inv (X - LF))
    """   
    n_idx = len(idx)
    n_time = facs.shape[1]

    if skip_batching:

        # Aggregate data and inverse error covariance
        data_list = []
        inv_err_cov_list = []
        for data, inv_err_cov in zip(get_data_loader(), get_inv_err_cov_loader()):
            data_list.append(data)
            inv_err_cov_list.append(inv_err_cov)
        data = np.concatenate(data_list)
        inv_err_cov = np.concatenate(inv_err_cov_list)

        # Compute loss
        errs = data[idx] - loads[:,idx].T @ facs
        out = np.trace(errs.T @ inv_err_cov[idx,:][:,idx] @ errs)

    else:

        out = 0
        start_1 = 0
        for data_1, inv_err_cov_1 in zip(get_data_loader(), get_inv_err_cov_loader()):

            sz_1 = len(data_1)
            idx_1_mask = np.logical_and(idx >= start_1, idx < start_1 + sz_1)
            idx_1 = idx[idx_1_mask]

            errs_1 = data_1[idx_1 - start_1] - loads.T[idx_1] @ facs
            inv_err_cov_1 = inv_err_cov_1[idx_1 - start_1]

            start_2 = 0
            for data_2 in get_data_loader():

                sz_2 = len(data_2)
                idx_2_mask = np.logical_and(idx >= start_2, idx < start_2 + sz_2)
                idx_2 = idx[idx_2_mask]

                errs_2 = data_2[idx_2 - start_2] - loads.T[idx_2] @ facs
                inv_err_cov_1_2 = inv_err_cov_1[:,idx_2]

                out += np.trace(errs_1.T @ inv_err_cov_1_2 @ errs_2)
                start_2 += sz_2
            
            start_1 += sz_1
        
    return out / n_idx / n_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--est_method', type=str, choices=['rbels', 'rbegls'])
    parser.add_argument('--n_time', type=int)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--skip_batching', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    # Directory/path preparation
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    path_gamma = os.path.join(args.dir_out, f'gamma-{args.est_method}-r{args.regime}.pt')
    path_out_facs = os.path.join(args.dir_out, f'facs_split-train_reg-{args.regime}_{args.est_method}.pt')  # estimate_factor_scores.py output path
    path_best_facs = os.path.join(args.dir_out, f'facs_split-train_reg-{args.regime}_{args.est_method}_best.pt')
    
    # Get loadings
    if args.regime in [1, 2]: 
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path).numpy()
        if args.path_mask is not None: 
            mask = torch.flatten(torch.load(args.path_mask)).numpy()
            loads = loads[:,mask]
    else: 
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-full.pth')
        loads = torch.load(path)['loads'].t().numpy()

    # Get validation indices
    path = os.path.join(dir_data, 'idx-space_split-valid_.pt')
    idx_valid = torch.load(path).numpy()

    # Create dataloader for validation data
    get_data_loader = partial(
        get_generator,
        gen_fcn=gen_tensors_as_arrays,
        dir=dir_data,
        prefix=f'data-space_split-full',
        batch_size=args.batch_size,
        sort_by=('i', int)
    )
    if args.est_method == 'rbegls':
        get_inv_err_cov_loader = partial(
            get_generator,
            gen_fcn=gen_arrays,
            dir=os.path.join(args.dir_out, 'err-cov'),
            prefix=f'inv-err-cov_reg-{args.regime}_',
            batch_size=args.batch_size,
            sort_by=('i', int)
        )

    # Build path/flags
    path = os.path.join(config.root, f'estimate_factor_scores.py')
    flags = {
        'config': args.config,
        'dir_out': args.dir_out,
        'dir_out_scratch': args.dir_out_scratch,
        'est_methods': args.est_method,
        'split': 'train',
        'est_method_loads': args.est_method_loads,
        'regime': args.regime,
        'batch_size': args.batch_size 
    }
    if args.path_mask is not None:
        flags['path_mask'] = args.path_mask

    # Estimate model for various gammas
    results = pd.DataFrame(columns=['gamma', 'valid_loss'])
    best_gamma = None
    best_valid_loss = float('inf')
    for gamma in GAMMAS: 
        flags['gamma'] = gamma
        code = execute_script(path, flags, False)
        if code != 0:
            sys.exit(code)

        # Read in training factors
        facs = torch.load(path_out_facs).numpy()

        # Compute LS loss
        if args.est_method == 'rbels':
            valid_loss = compute_ls_loss(
                idx_valid, 
                get_data_loader, 
                loads, facs, 
                skip_batching=args.skip_batching
            )
        if args.est_method == 'rbegls':
            valid_loss = compute_gls_loss(
                idx_valid,
                get_data_loader, get_inv_err_cov_loader,
                loads, facs, 
                skip_batching=args.skip_batching
            )
        print(f"gamma = {gamma} | valid_loss = {valid_loss}")

        # Add row to dataframe
        row = {'gamma': gamma, 'valid_loss': valid_loss}
        results = pd.concat([results, pd.DataFrame([row])])

        # Break if improvement stops or update best model
        if valid_loss > best_valid_loss:
            break
        best_valid_loss = valid_loss
        best_gamma = gamma
        os.rename(path_out_facs, path_best_facs)

    # Save best gamma and train factors
    torch.save(torch.tensor(best_gamma, dtype=torch.float64), path_gamma)
    os.replace(path_best_facs, path_out_facs)

    # Write dataframe of results to file
    path = os.path.join(args.dir_out, f'gamma-results-{args.est_method}.csv')
    results.to_csv(path, index=False)

