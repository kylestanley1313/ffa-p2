import argparse
import os
import sys
import torch
from functools import partial

from utils import (
    compute_loss,
    execute_script, 
    load_config,
    model_from_loads,
)


def compute_cv_loss_for_sigmas(
        sigmas, 
        n_folds,
        config,
        dir_out,
        dir_out_scratch,
        est_method,
        rot_method,
        rot_mats,
        sz_space,
        path_mask=None
    ):

    # Path preparation
    dir_cov = os.path.join(dir_out_scratch, f'cov')
    path_out = lambda v: os.path.join(args.dir_out, f'model-{est_method}-train-{v}-{rot_method}-smooth.pth')

    # Build Execution parameters
    path_script = os.path.join(config.root, 'scripts', 'smooth_loads.py')
    flags = {
        'config': args.config,
        'dir_out': dir_out,
        'sigmas': sigmas,
        'sz_space': sz_space,
        'split': 'train',
        'est_method': est_method,
        'rot_method': rot_method,
    }
    if args.path_mask is not None: 
        flags['path_mask'] = path_mask

    loss = 0
    for fold in range(n_folds):

        # Smooth training model
        flags['fold'] = fold
        code = execute_script(path_script, flags, False)
        if code != 0: 
            sys.exit(code)

        # Compute validation loss
        loads = torch.load(path_out(fold))['loads']
        model = model_from_loads(loads @ torch.linalg.inv(rot_mats[fold]))
        loss += compute_loss(model, dir_cov, 'valid', fold).item() / n_folds

        # Clean up
        if os.path.exists(path_out(fold)):
            os.remove(path_out(fold))

    return loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--est_method', type=str)
    parser.add_argument('--rot_method', type=str)
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument(
        '--sigma_grid', type=float, nargs='+',
        default=[0, 0.35, 0.40, 0.45, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
        help=(
            "Should start with zero (no smoothing). "
            "Second element should be minimal smoothing."
        )
    )
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--universal_param', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    # Read in rotation matrices
    rot_mats = [None] * args.n_folds
    for v in range(args.n_folds): 
        path = os.path.join(args.dir_out, f'rot-{args.rot_method}-train-{v}.pt')
        rot_mats[v] = torch.load(path)

    compute_cv_loss_for_sigmas_ = partial(
        compute_cv_loss_for_sigmas,
        n_folds=args.n_folds,
        config=config,
        dir_out=args.dir_out,
        dir_out_scratch=args.dir_out_scratch,
        est_method=args.est_method,
        rot_method=args.rot_method,
        rot_mats=rot_mats,
        path_mask=args.path_mask,
        sz_space=args.sz_space
    )

    # Initialize loss
    n_facs = torch.load(
        os.path.join(args.dir_out, f'model-{args.est_method}-train-0-{args.rot_method}.pth')
    )['loads'].shape[1]
    best_idx = [0] * n_facs 
    max_idx = len(args.sigma_grid) - 1
    sigmas = [args.sigma_grid[_] for _ in best_idx]
    best_loss = compute_cv_loss_for_sigmas_(sigmas)
    print(f"sigmas = {sigmas} | valid_loss = {best_loss}", flush=True)
    

    # Tune sigmas via universal descent
    if args.universal_param: 

        # Set index deltas
        deltas = torch.arange(1, args.radius + 1)

        for i in range(1, args.max_iters + 1):

            best_loss_prev = best_loss
                
            # Create candidate indices
            indices = deltas + best_idx[0]
            indices = indices[indices <= max_idx].tolist()

            for idx in indices: 
                
                # Compute validation loss
                test_idx = torch.tensor([idx] * n_facs, dtype=torch.int32)
                sigmas = [args.sigma_grid[_] for _ in test_idx]
                valid_loss = compute_cv_loss_for_sigmas_(sigmas)
                print(f"sigmas = {sigmas} | valid_loss = {valid_loss}", flush=True)

                # Update bests
                if valid_loss < best_loss:
                    best_idx = [idx] * n_facs
                    best_loss = valid_loss

            print(f"iter = {i} | loss = {best_loss}", flush=True)
            if best_loss_prev - best_loss < args.tol: 
                break
            best_loss_prev = best_loss

            
    # Tune sigmas via coordinate descent
    else:

        # Set index deltas
        tmp1 = -torch.flip(torch.arange(1, args.radius + 1), dims=[0])
        tmp2 = torch.arange(1, args.radius + 1)
        deltas = torch.concatenate((tmp1, tmp2))

        for i in range(1, args.max_iters + 1):

            best_loss_prev = best_loss

            for k in range(n_facs):

                best_idx_k = best_idx[k]
                best_loss_k = best_loss

                # Create candidate indices
                indices = deltas + best_idx[k]
                mask = torch.logical_and(indices >= 0, indices <= max_idx)
                indices = indices[mask].tolist()

                for idx in indices: 

                    # Compute validation loss
                    test_idx = best_idx.copy()
                    test_idx[k] = idx
                    sigmas = [args.sigma_grid[_] for _ in test_idx]
                    valid_loss = compute_cv_loss_for_sigmas_(sigmas)
                    print(f"sigmas = {sigmas} | valid_loss = {valid_loss}", flush=True)

                    # Update bests
                    if valid_loss < best_loss_k:
                        best_idx_k = idx
                        best_loss_k = valid_loss
                
                # Update bests
                best_idx[k] = best_idx_k
                best_loss = best_loss_k

            print(f"iter = {i} | loss = {best_loss}", flush=True)
            if best_loss_prev - best_loss < args.tol: 
                break
            best_loss_prev = best_loss


    # Save sigmas
    path = os.path.join(args.dir_out, f'sigmas-{args.est_method}-{args.rot_method}.pt')
    sigmas = torch.tensor([args.sigma_grid[_] for _ in best_idx])
    torch.save(sigmas, path)



