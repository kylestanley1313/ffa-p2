import argparse
import os
import pandas as pd
import subprocess
import torch

from config import load_config
from utils import model_from_loads, refresh_directory, remove_directory



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument(
        '--rot_method', type=str, 
        choices=['varimax', 'quartimin'],
        required=True
    )
    parser.add_argument('--file_trg', type=str)
    parser.add_argument('--est_method', type=str)
    parser.add_argument('--split', type=str, choices=['full', 'train'])
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Commandline argument parsing
    target = True if args.file_trg else False
    split_str = 'full' if args.split == 'full' else f'{args.split}-{args.fold}'

    # Path preparation
    dir_tmp = os.path.join(args.dir_out, 'tmp-rot')
    refresh_directory(dir_tmp)
    path_in = os.path.join(args.dir_out, f'model-{args.est_method}-{split_str}.pth') 
    path_in_csv = os.path.join(dir_tmp, f'loads-{args.est_method}-{split_str}.csv.gz') 
    path_rot = os.path.join(args.dir_out, f'rot-{args.rot_method}-{split_str}.pt')
    path_rot_csv = os.path.join(dir_tmp, f'rot-{args.rot_method}-{split_str}.csv.gz')
    path_out = os.path.join(args.dir_out, f'model-{args.est_method}-{split_str}-{args.rot_method}.pth') 
    path_out_csv = os.path.join(dir_tmp, f'loads-{args.est_method}-{split_str}-{args.rot_method}.csv.gz')
    if target: # target is always full roated loadings
        path_trg = os.path.join(args.dir_out, args.file_trg)
        path_trg_csv = os.path.join(args.dir_out, dir_tmp, args.file_trg.replace('model-', 'loads-').replace('.pth', '.csv.gz'))

    # Read loadings from .pth
    loads = pd.DataFrame(torch.load(path_in)['loads'].numpy())
    loads.to_csv(path_in_csv, header=False, index=False)
    if target:
        loads_trg = pd.DataFrame(torch.load(path_trg)['loads'].numpy())
        loads_trg.to_csv(path_trg_csv, header=False, index=False)

    # Call R script (rotates then writes to .csv)
    kwargs = {
        'path_in': path_in_csv,
        'path_out': path_out_csv,
        'path_rot': path_rot_csv,
        'rot_method': args.rot_method,
    }
    if target:
        kwargs['path_trg'] = path_trg_csv
    command = ['Rscript', f'{config.root}/rotate.R'] + [f"--{k}={v}" for k, v in kwargs.items()]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("R script output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("R script error:\n", e.stderr)

    # Read rotation and rotated loadings from .csv
    loads_rot = torch.from_numpy(
        pd.read_csv(path_out_csv, header=None, index_col=None).values
    ).to(torch.float32)
    rot_mat = torch.from_numpy(
        pd.read_csv(path_rot_csv, header=None, index_col=None).values
    ).to(torch.float32)

    # Write rotated loadings to .pth
    model = model_from_loads(loads_rot)
    torch.save(model.state_dict(), path_out)
    torch.save(rot_mat, path_rot)

    # Clean up CSVs
    remove_directory(dir_tmp)

