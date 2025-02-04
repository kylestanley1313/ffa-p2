import argparse
import numpy as np
import os
import pandas as pd
import torch

from config import load_config
from utils import (
    compute_loss, 
    execute_script, 
    model_from_loads, 
    multiply_list,
)


# TODO: Refine this grid
SIGMAS = [0, 0.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--n_sub', type=int)
    parser.add_argument('--n_comps', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Directory/path preparation
    dir_cov = os.path.join(args.dir_out_scratch, 'cov')
    path_sigma = os.path.join(args.dir_out, f'sigma.pt')
    path_out_space = os.path.join(args.dir_out, 'ica-space_split-train.npy')
    path_out_time = os.path.join(args.dir_out, 'ica-time_split-train.npy')

    # Build path/flags
    path = os.path.join(config.root, 'melodic_estimation.py')
    flags = {
        'config': args.config,
        'dir_out': args.dir_out,
        'dir_out_scratch': args.dir_out_scratch,
        'split': 'train',
        'n_sub': args.n_sub,
        'n_comps': args.n_comps
    }

    # Run MELODIC for various sigmas
    best_sigma = None
    best_valid_loss = float('inf')
    for sigma in SIGMAS:

        # Run MELODIC
        flags['sigma'] = sigma
        code = execute_script(path, flags, False)

        # Compute validation loss
        loads = np.load(path_out_space)
        loads = loads.reshape(multiply_list(loads.shape[:3]), loads.shape[3])
        model = model_from_loads(torch.tensor(loads))
        valid_loss = compute_loss(model, dir_cov, 'valid').item()
        print(f"sigma = {sigma} | valid_loss = {valid_loss}")

        # Break from loop if improvement stops or update best model
        if valid_loss > best_valid_loss:
            break
        best_valid_loss = valid_loss
        best_sigma = sigma

    # Save best sigma
    torch.save(torch.tensor(best_sigma, dtype=torch.float32), path_sigma)

    # Cleanup models
    if os.path.exists(path_out_space):
        os.remove(path_out_space)
    if os.path.exists(path_out_time):
        os.remove(path_out_time)


