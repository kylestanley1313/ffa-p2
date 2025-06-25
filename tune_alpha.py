import argparse
import os
import pandas as pd
import sys
import torch

from config import load_config
from utils import (
    compute_loss,
    execute_script, 
    model_from_loads,
)


ALPHAS = [0, 0.001, 0.01, 0.1, 1, 10]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--est_method', type=str, choices=['lbfgs', 'dsgd'])
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument('--n_facs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--history_size', type=int)
    parser.add_argument('--tol', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    # Validate command-line arguments
    if args.est_method == 'lbfgs':
        assert args.history_size is not None, "Must pass history size!"
    if args.est_method == 'dsgd':
        assert args.world_size is not None, "Must pass world size!"
        assert args.batch_size is not None, "Must pass batch size!"
        assert args.seed is not None, "Must pass seed!"

    config = load_config(args.config)

    # Directory/path preparation
    dir_cov = os.path.join(args.dir_out_scratch, f'cov')
    path_alpha = os.path.join(args.dir_out, f'alpha-{args.est_method}.pt')
    path_out_mod = os.path.join(args.dir_out, f'model-{args.est_method}-train.pth')  # estimation.py output path
    path_best_mod = os.path.join(args.dir_out, f'model-{args.est_method}-train-best.pth')

    # Build path/flags
    path = os.path.join(config.root, f'estimate_loads_{args.est_method}.py')
    flags = {
        'config': args.config,
        'dir_out': args.dir_out,
        'dir_out_scratch': args.dir_out_scratch,
        'split': 'train',
        'path_mask': args.path_mask,
        'sz_space': args.sz_space,
        'n_facs': args.n_facs,
        'lr': args.lr,
        'tol': args.tol,
        'patience': args.patience,
        'max_epochs': args.max_epochs,
    }
    if args.est_method == 'lbfgs':
        flags['history_size'] = args.history_size
    if args.est_method == 'dsgd': 
        flags['world_size'] = args.world_size
        flags['batch_size'] = args.batch_size
        flags['seed'] = args.seed
    if args.benchmark:
        flags['benchmark'] = None

    # Estimate model for various alphas
    results = pd.DataFrame(columns=['alpha', 'valid_loss'])
    best_alpha = None
    best_valid_loss = float('inf')
    for alpha in ALPHAS:
        flags['alpha'] = alpha
        code = execute_script(path, flags, False)
        if code != 0: 
            sys.exit(code)

        # Compute validation loss
        loads = torch.load(path_out_mod)['loads']
        model = model_from_loads(loads)
        valid_loss = compute_loss(model, dir_cov, 'valid').item()
        print(f"alpha = {alpha} | valid_loss = {valid_loss}")
        
        # Add row to dataframe
        row = {'alpha': alpha, 'valid_loss': valid_loss}
        results = pd.concat([results, pd.DataFrame([row])])

        # Break from loop if improvement stops or update best model
        if valid_loss > best_valid_loss:
            break
        best_valid_loss = valid_loss
        best_alpha = alpha
        os.rename(path_out_mod, path_best_mod)

    # Save best alpha and model-train
    # NOTE: If improvement occurs for each successive alpha, the last one will
    # be saved.
    torch.save(torch.tensor(best_alpha, dtype=torch.float64), path_alpha)
    os.replace(path_best_mod, path_out_mod)

    # Write dataframe of results to file
    path = os.path.join(args.dir_out, f'alpha-results-{args.est_method}.csv')
    results.to_csv(path, index=False)
