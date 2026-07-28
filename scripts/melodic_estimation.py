import argparse
import nibabel as nib
import numpy as np
import os
import shutil
import subprocess
import sys
import torch

from typing import Optional

from utils import load_config



def run_subprocess(args):
    proc = subprocess.run(args, capture_output=True, check=True, text=True)
    if len(proc.stderr) > 0:
        print(proc.stderr, file=sys.stderr)
        return 1
    else: 
        print(proc.stdout)
        return 0



def run(
        dir_out: str, 
        dir_out_scratch: str,
        fold: int, 
        n_folds: int,
        sigma: Optional[float], 
        n_comps: int
    ) -> None:

    # Create directories
    dir_data = os.path.join(dir_out_scratch, 'data')
    dir_data_ica = os.path.join(dir_out_scratch, 'data-ica')
    dir_ica = os.path.join(dir_out_scratch, 'ica')
    os.makedirs(dir_ica, exist_ok=True)

    # Path constructors
    path_data = lambda n: os.path.join(dir_data_ica, f'data_split-full_n-{n}.nii.gz')
    path_smooth = lambda n: os.path.join(dir_data_ica, f'data_split-full_n-{n}_smooth.nii.gz')

    # Get subjects
    subs = []
    for v in range(n_folds):
        if v == fold:
            continue
        path = os.path.join(dir_data, f'subs_v-{v}.pt')
        subs += torch.load(path).tolist()

    # Smooth scans
    if sigma is not None and sigma > 0: 
        for n in subs:
            run_subprocess(['fslmaths', path_data(n), '-s', str(sigma), path_smooth(n)])
        input_files = [path_smooth(n) for n in subs]
    else: 
        input_files = [path_data(n) for n in subs]

    # Run MELODIC
    args = [
        'melodic',
        '-i', ','.join(input_files),
        '-o', dir_ica,
        '--nomask',
        '--nl=pow3',
        '--nobet',
        '--tr=1.0',
        '--Oorig',
        # '--Opca',
        '--disableMigp',
        '--varnorm',
        '--maxit=1000',
        '-d', str(n_comps),
        '--seed=12345'
    ]
    code = run_subprocess(args)

    # Handle MELODIC error
    if code != 0:
        shutil.rmtree(dir_ica) 
        sys.exit(code)

    # Extract results
    # NOTE: These MELODIC outputs appear to be on the same scale as the true 
    # loadings and factors, resp.
    path = os.path.join(dir_ica, 'melodic_oIC.nii.gz')
    # path = os.path.join(dir_ica, 'melodic_pca.nii.gz')
    comps = nib.load(path).get_fdata().astype(np.float32)
    path = os.path.join(dir_ica, 'melodic_mix')
    mix_mat = np.loadtxt(path).astype(np.float32)

    # Write results
    method = 'ica' if sigma is None else 'icas' 
    path = os.path.join(
        dir_out, 
        f'{method}-space_split-full.npy' if fold is None else f'{method}-space_split-train_v-{fold}.npy'
    )
    np.save(path, comps)
    path = os.path.join(
        dir_out, 
        f'{method}-time_split-full.npy' if fold is None else f'{method}-time_split-train_v-{fold}.npy'
    )
    np.save(path, mix_mat)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--n_comps', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    run(
        args.dir_out, args.dir_out_scratch, 
        args.fold, args.n_folds,
        args.sigma, args.n_comps
    )


   