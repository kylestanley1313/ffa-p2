import argparse
import os
import torch

from config import load_config
from utils.utils import (
    compute_loss,
    execute_script, 
    loss_fcn, 
    model_from_loads,
    read_tensors
)
from utils.model import LowRankCovariance


ALPHAS = [0, 1, 10] # , 100, 1000, 10000]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
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
    dir_cov = os.path.join(args.dir_out_scratch, 'cov')
    path_alpha = os.path.join(args.dir_out, 'alpha.pt')
    path_out_mod = os.path.join(args.dir_out, 'model-train.pth')  # estimation.py output path
    path_best_mod = os.path.join(args.dir_out, 'model-train-best.pth')

    # Estimate model for various alphas
    best_alpha = None
    best_valid_loss = float('inf')
    for alpha in ALPHAS:

        # Estimate model with current alpha
        path = os.path.join(config.root, f'factor_model_ddp.py')
        flags = {
            'config': args.config,
            'dir_out': args.dir_out,
            'dir_out_scratch': args.dir_out_scratch,
            'world_size': args.world_size,
            'split': 'train',
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

        # Compute validation loss
        loads = torch.load(path_out_mod)['loads']
        model = model_from_loads(loads)
        valid_loss = compute_loss(model, dir_cov, 'valid')
        print(f"alpha = {alpha} | valid_loss = {valid_loss}")

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
