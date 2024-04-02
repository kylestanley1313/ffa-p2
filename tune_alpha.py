import argparse
import os
import torch

from config import load_config
from utils.utils import execute_script, loss_fcn, read_tensors, refresh_directory
from utils.model import LowRankCovariance


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
    dir_cov = os.path.join(dir_out, 'cov')
    path_alpha = os.path.join(dir_out, 'alpha.pt')
    path_out_mod = os.path.join(dir_out, 'model-train.pth')  # estimation.py output path
    path_best_mod = os.path.join(dir_out, 'model-train-best.pth')

    # Estimate model for various alphas
    best_alpha = None
    best_valid_loss = float('inf')
    for alpha in ALPHAS:

        print('HERE')

        # Estimate model with current alpha
        path = os.path.join(config.root, f'factor_model_ddp.py')
        flags = {
            'config': args.config,
            'dir_out': args.dir_out,
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
        model = LowRankCovariance(loads.shape[0], loads.shape[1])
        model.set_loads(loads)
        points_loader = read_tensors(dir_cov, 'points')
        cov_valid_loader = read_tensors(dir_cov, 'cov-valid')
        valid_loss = 0
        for points, cov_valid in zip(points_loader, cov_valid_loader):
            preds = model(points)
            valid_loss += loss_fcn(preds, cov_valid, loads.shape[0])
        print(f"alpha = {alpha} | valid_loss = {valid_loss}")

        # Update best model
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_alpha = alpha
            os.rename(path_out_mod, path_best_mod)
        else: 
            torch.save(torch.tensor(best_alpha, dtype=torch.float64), path_alpha)
            os.rename(path_best_mod, path_out_mod)
            break

