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
    gen_tensors_as_arrays,
)


GAMMAS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]  # TODO: Set gammas


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--dir_truth', type=str)
    parser.add_argument('--est_method', type=str, choices=['rbels', 'rbegls'])
    parser.add_argument('--n_time', type=int)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Directory/path preparation
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    path_gamma = os.path.join(args.dir_out, f'gamma-{args.est_method}-r{args.regime}.pt')
    path_out_facs = os.path.join(args.dir_out, f'facs_train_r{args.regime}_{args.est_method}.pt')  # estimate_factor_scores.py output path
    path_best_facs = os.path.join(args.dir_out, f'facs_train_r{args.regime}_{args.est_method}_best.pt')
    
    # Get loadings
    if args.regime == 1: 
        path = os.path.join(args.dir_truth, 'loads.pt')
        loads = torch.load(path).numpy()
    else: 
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-full.pth')
        loads = torch.load(path)['loads'].t().numpy()

    # Get validation indices
    path = os.path.join(dir_data, 'idx-space-valid.pt')
    idx_valid = torch.load(path).numpy()

    # Create dataloader for validation data
    get_data_loader = partial(
        get_generator,
        gen_fcn=gen_tensors_as_arrays,
        dir=dir_data,
        prefix=f'data-space-full',
        batch_size=args.batch_size
    )

    # Build path/flags
    path = os.path.join(config.root, f'estimate_factor_scores.py')
    flags = {
        'config': args.config,
        'dir_out': args.dir_out,
        'dir_out_scratch': args.dir_out_scratch,
        'dir_truth': args.dir_truth,
        'est_methods': args.est_method,
        'split': 'train',
        'est_method_loads': args.est_method_loads,
        'regime': args.regime,
        'batch_size': args.batch_size 
    }

    # Estimate model for various gammas
    results = pd.DataFrame(columns=['gamma', 'valid_loss'])
    best_gamma = None
    best_valid_loss = float('inf')
    for gamma in GAMMAS: 
        flags['gamma'] = gamma
        code = execute_script(path, flags, False)  # TODO: Revise exit code behavior?
        if code != 0:
            sys.exit(code)

        # Compute validation loss
        facs = torch.load(path_out_facs).numpy()

        valid_loss = 0
        start = 0
        for data in get_data_loader():
            sz = len(data)
            idx_mask = np.logical_and(idx_valid >= start, idx_valid < start + sz)
            idx_ = idx_valid[idx_mask]
            preds = loads[:,idx_].T @ facs
            valid_loss += np.sum((preds - data[idx_ - start]) ** 2) / len(idx_valid)
            start += sz
        print(f"gamma = {gamma} | valid_loss = {valid_loss}")

        # Add row to dataframe
        row = {'gamma': gamma, 'valid_loss': valid_loss}
        results = pd.concat([results, pd.DataFrame([row])])

        # Break from loop if improvement stops or update best model
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

