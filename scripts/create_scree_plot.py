import argparse
import csv
import os
import torch

from datetime import datetime

from config import load_config
from utils import (
    compute_loss,
    execute_script,
    model_from_loads,
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('--min_n_facs', type=int)
    parser.add_argument('--max_n_facs', type=int)
    parser.add_argument('--prop_init', type=float, default=1.0)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Set paths
    path_init = os.path.join(args.dir_out, 'init-loads-full.pt')
    path_model = os.path.join(args.dir_out, 'model-dssgd-full.pth')
    dir_cov = os.path.join(args.dir_out, 'cov')

    # Compute variable count
    path = os.path.join(
        args.dir_out_scratch, 'data',
        'data-time_split-full_n-0_i-0_.pt'
    )
    n_vars = torch.load(path).shape[1]

    # Create CSV in which to store results
    path_csv = os.path.join(args.dir_out, 'scree_plot.csv')
    with open(path_csv, 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(['k', 'loss'])

    for k in range(args.min_n_facs, args.max_n_facs + 1):
        print(f"\n========== K = {k} ==========", flush=True)
    
        # Initialize loadings
        print("----- Loading Initialization -----", flush=True)
        path = os.path.join(config.root, f'initialize_loadings.py')
        flags = {
            'config': args.config,
            'dir_out': args.dir_out,
            'dir_out_scratch': args.dir_out_scratch,
            'split': 'full',
            'n_folds': args.n_folds,
            'n_facs': k,
            'init_method': 'pca_randomized',
            'prop_init': args.prop_init,
            'seed': args.seed,
        }
        execute_script(path, flags)

        # Estimate loadings (TODO: after covariance computed)
        print("----- Loading Estimation -----", flush=True)
        path = os.path.join(config.root, f'estimate_loads_dssgd.py')
        flags = { # TODO: SGD parameters may need tuning
            'config': args.config,
            'dir_out': args.dir_out,
            'dir_out_scratch': args.dir_out_scratch,
            'split': 'full',
            'n_facs': k,
            'n_vars': n_vars,
            'world_size': args.world_size,
            'batch_size': 4096,
            'lr': 16.0,
            'tol': 1e-7,
            'patience': 5,
            'max_epochs': 1000,
            'seed': args.seed
        }
        execute_script(path, flags)

        # Compute loss (TODO: after covariance computed)
        print("----- Loss Computation -----", flush=True)
        loads = torch.load(path_model)['loads']
        # loads = torch.load(path_init)
        model = model_from_loads(loads)
        loss = compute_loss(model, dir_cov, 'full')
        with open(path_csv, 'a', newline='') as file: 
            writer = csv.writer(file)
            writer.writerow([k, loss.item()])
            print(f"[{datetime.now()}] k = {k} | loss = {loss.item()}")

        # Clean up
        try: 
            os.remove(path_init)
            os.remove(path_model)
        except FileNotFoundError:
            pass

