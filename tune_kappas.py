import argparse
import os
import sys
import torch
from functools import partial

from config import load_config
from utils import (
    compute_loss,
    execute_script, 
    model_from_loads,
)



def compute_cv_loss_for_kappas(
        kappas, 
        n_folds,
        config,
        dir_out,
        dir_out_scratch,
        est_method,
        rot_method,
        rot_mats
    ):

    # Path preparation
    dir_cov = os.path.join(dir_out_scratch, f'cov')
    path_out = lambda v: os.path.join(
        dir_out, 
        f'model-{est_method}-train-{v}-{rot_method}-smooth-shrink.pth'
        # f'model-{est_method}-train-{v}-{rot_method}-shrink.pth'
    )

    # Build Execution parameters
    path_script = os.path.join(config.root, 'shrink_loads.py')
    flags = {
        'config': config.name,
        'dir_out': dir_out,
        'kappas': kappas,
        'split': 'train',
        'est_method': est_method,
        'rot_method': rot_method,
    }

    loss = 0
    for fold in range(n_folds):

        # Shrink training model
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
    parser.add_argument('--n_folds', type=int)
    parser.add_argument(
        '--method', type=str,
        choices=['universal', 'coordinate', 'sequential']
    )
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--radius', type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)

    # Path preparation
    path_in = lambda v: os.path.join(
        args.dir_out, 
        f'model-{args.est_method}-train-{v}-{args.rot_method}-smooth.pth'
    )

    # Define kappa grid
    max_load = 0
    for v in range(args.n_folds):
        loads = torch.load(path_in(v))['loads'].t()
        max_load = max(max_load, torch.max(torch.abs(loads)).item())
    kappa_grid = torch.arange(0, max_load + 0.01, 0.01) ** 3

    # Read in rotation matrices
    rot_mats = [None] * args.n_folds
    for v in range(args.n_folds): 
        path = os.path.join(args.dir_out, f'rot-{args.rot_method}-train-{v}.pt')
        rot_mats[v] = torch.load(path)

    # Define CV loss function
    compute_cv_loss_for_kappas_ = partial(
        compute_cv_loss_for_kappas,
        n_folds=args.n_folds,
        config=config,
        dir_out=args.dir_out,
        dir_out_scratch=args.dir_out_scratch,
        est_method=args.est_method,
        rot_method=args.rot_method,
        rot_mats=rot_mats
    )
    
    # Initialize loss
    n_facs = torch.load(path_in(0))['loads'].shape[1]
    best_idx = [0] * n_facs
    max_idx = len(kappa_grid) - 1
    kappas = [kappa_grid[_].item() for _ in best_idx]
    best_loss = compute_cv_loss_for_kappas_(kappas)
    print(f"kappas = {kappas} | valid_loss = {best_loss}", flush=True)

    # Tune kappas via universal descent
    if args.method == 'universal':
        
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
                kappas = [kappa_grid[_].item() for _ in test_idx]
                valid_loss = compute_cv_loss_for_kappas_(kappas)
                print(f"kappas = {kappas} | valid_loss = {valid_loss}", flush=True)

                # Update bests
                if valid_loss < best_loss:
                    best_idx = [idx] * n_facs
                    best_loss = valid_loss

            print(f"iter = {i} | loss = {best_loss}", flush=True)
            if best_loss_prev - best_loss < args.tol: 
                break
            best_loss_prev = best_loss


    # Tune kappas via coordinate descent
    elif args.method == 'coordinate': 

        # Set index deltas
        tmp1 = -torch.flip(torch.arange(1, args.radius + 1), dims=[0])
        tmp2 = torch.arange(1, args.radius + 1)
        deltas = torch.concatenate((tmp1, tmp2))

        for i in range(1, args.max_iters + 1): 

            best_loss_prev = best_loss

            for k in range(n_facs):

                best_idx_k = best_idx[k]
                best_loss_k = best_loss

                # Create canddiate indices
                idx = deltas + best_idx[k]
                mask = torch.logical_and(idx >= 0, idx <= max_idx)
                idx = idx[mask].tolist()

                for idx_ in idx: 

                    # Compute validation loss
                    test_idx = best_idx.copy()
                    test_idx[k] = idx_
                    kappas = [kappa_grid[_].item() for _ in test_idx]
                    valid_loss = compute_cv_loss_for_kappas_(kappas)
                    print(f"kappas = {kappas} | valid_loss = {valid_loss}", flush=True)

                    # Update bests
                    if valid_loss < best_loss_k:
                        best_idx_k = idx_
                        best_loss_k = valid_loss

                # Update bests
                best_idx[k] = best_idx_k
                best_loss = best_loss_k

            print(f"iter = {i} | loss = {best_loss}")
            if best_loss_prev - best_loss < args.tol: 
                break
            best_loss_prev = best_loss

    elif args.method == 'sequential':
        
        for k in range(n_facs):
            print(f"Tuning k = {k}...\n")
            
            test_idx = best_idx.copy()
            last_loss = best_loss

            for idx in range(1, max_idx + 1):
                
                test_idx[k] = idx
                kappas = [kappa_grid[_].item() for _ in test_idx]
                loss = compute_cv_loss_for_kappas_(kappas)
                print(f"kappas = {kappas} | valid_loss = {loss}", flush=True)

                if loss >= last_loss:  # if loss increases or load[k] zero'd out one step before
                    best_idx[k] = idx - 1
                    best_loss = last_loss
                    break
            
                if idx == max_idx:  # if maximum kappa achieved
                    best_idx[k] = idx
                    best_loss = last_loss
                    break

                last_loss = loss

    else: 
        raise Exception("Invalid tuning method!")

    # Save kappas
    path = os.path.join(args.dir_out, f'kappas-{args.est_method}-{args.rot_method}.pt')
    kappas = torch.tensor([kappa_grid[_] for _ in best_idx])
    torch.save(kappas, path)




