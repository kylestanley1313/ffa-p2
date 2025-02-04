import argparse
import nibabel as nib
import numpy as np
import os
import subprocess
import torch

from config import load_config



def run_subprocess(args):
    try:
        proc = subprocess.run(args, capture_output=True, check=True, text=True)
        print(proc.stdout)
    except subprocess.CalledProcessError as err: 
        print(f"returncode = {err.returncode} "
              f"stderr = {err.stderr} "
              f"stdout = {err.stdout}")
        # TODO: Add exit code


def run(
        dir_out: str, 
        dir_out_scratch: str,
        split: str, 
        n_sub: int, 
        sigma: float, 
        n_comps: int
    ) -> None:

    # Create directories
    dir_data_ica = os.path.join(dir_out_scratch, 'data-ica')
    dir_ica = os.path.join(dir_out_scratch, 'ica')
    os.makedirs(dir_ica, exist_ok=True)

    # Path constructors
    path_data = lambda n: os.path.join(dir_data_ica, f'data_split-{split}_n-{n}.nii.gz')
    path_smooth = lambda n: os.path.join(dir_data_ica, f'data_n-{n}_smooth.nii.gz')

    # Smooth scans
    if sigma > 0: 
        for n in range(n_sub):
            run_subprocess(['fslmaths', path_data(n), '-s', str(sigma), path_smooth(n)])
        input_files = [path_smooth(n) for n in range(n_sub)]
    else: 
        input_files = [path_data(n) for n in range(n_sub)]

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
        '--disableMigp',
        '--varnorm',
        '--maxit=1000',
        '-d', str(n_comps),
        '--seed=12345'
    ]
    run_subprocess(args)

    # Extract results
    # NOTE: These MELODIC outputs appear to be on the same scale as the true 
    # loadings and factors, resp.
    path = os.path.join(dir_ica, 'melodic_oIC.nii.gz')
    comps = nib.load(path).get_fdata().astype(np.float32)
    path = os.path.join(dir_ica, 'melodic_mix')
    mix_mat = np.loadtxt(path).astype(np.float32)

    # Write results
    path = os.path.join(dir_out, f'ica-space_split-{split}.npy')
    np.save(path, comps)
    path = os.path.join(dir_out, f'ica-time_split-{split}.npy')
    np.save(path, mix_mat)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--n_sub', type=int)
    parser.add_argument('--sigma', type=float, default=0)
    parser.add_argument('--n_comps', type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    run(
        args.dir_out, args.dir_out_scratch, 
        args.split, args.n_sub, args.sigma, args.n_comps
    )


   