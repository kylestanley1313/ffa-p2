import argparse
import os
import torch

from config import load_config
from utils.utils import execute_script, refresh_directory


ALPHAS = [0, 1, 10, 100, 1000, 10000]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--grid_shape', type=int, nargs='+')
    parser.add_argument('--num_facs', type=int)
    parser.add_argument('--delta', type=float)
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized']
    )
    parser.add_argument('--init_prop', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--silent_fail', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    raise_error = not args.silent_fail

    # Directory/path preparation
    dir_out = os.path.join('out', args.dir_out)
    dir_out_tune = os.path.join(dir_out, 'tuning')
    refresh_directory(dir_out_tune)
    path_curr_mod = os.path.join(dir_out_tune, 'curr-model.pth')
    path_best_mod = os.path.join(dir_out_tune, 'best-model.pth')
    path_out_mod = os.path.join(dir_out, 'model.pth')  # estimation.py output path

    # Estimate model for various alphas
    best_alpha = None
    best_valid_loss = float('inf')
    for alpha in ALPHAS:

        # Estimate model with current alpha
        path = os.path.join(config.root, f'factor_model_ddp.py')
        flags = {
            'config': args.config,
            'dir_out': args.dir_out,
            'world_size': args.world_size,
            'grid_shape': args.grid_shape,
            'num_facs': args.num_facs,
            'alpha': alpha,
            'delta': args.delta,
            'init_method': args.init_method,
            'init_prop': args.init_prop,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_epochs': args.max_epochs
        }
        execute_script(path, flags, raise_error)

        # Update best model
        os.rename(path_out_mod, path_curr_mod)
        curr_model = torch.load(path_curr_mod)
        print(f"alpha = {alpha} | valid_loss = {curr_model['valid_loss']}")
        if curr_model['valid_loss'] <= best_valid_loss:
            best_valid_loss = curr_model['valid_loss']
            best_alpha = alpha
            os.rename(path_curr_mod, path_best_mod)
        else: 
            os.rename(path_best_mod, path_out_mod)
            break

