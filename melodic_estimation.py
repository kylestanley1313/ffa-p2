import argparse
import nibabel as nib
import numpy as np
import os
import subprocess
import torch

from typing import Optional

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
        '--seed=12346'
    ]
    run_subprocess(args)

    # Extract results
    # NOTE: These MELODIC outputs appear to be on the same scale as the true 
    # loadings and factors, resp.
    path = os.path.join(dir_ica, 'melodic_oIC.nii.gz')
    # path = os.path.join(dir_ica, 'melodic_pca.nii.gz')
    comps = nib.load(path).get_fdata().astype(np.float32)
    path = os.path.join(dir_ica, 'melodic_mix')
    mix_mat = np.loadtxt(path).astype(np.float32)

    # print(f"comps.shape = {comps.shape}")
    # path = os.path.join(dir_ica, 'melodic_pca.nii.gz')
    # tmp = nib.load(path).get_fdata().astype(np.float32)
    # print(f"tmp.shape = {tmp.shape}")



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


   