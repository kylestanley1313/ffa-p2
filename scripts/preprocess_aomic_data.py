import argparse
import os
import torch.multiprocessing as mp
import subprocess
from functools import partial



def run_subprocess(args):
    try:
        proc = subprocess.run(args, capture_output=True, check=True, text=True)
        print(proc.stdout)
    except subprocess.CalledProcessError as err: 
        print(f"returncode = {err.returncode} "
              f"stderr = {err.stderr} "
              f"stdout = {err.stdout}")


def process_sub_bet(sub: str, dir_in: str, dir_out: str) -> None:
    
    path_in = os.path.join(
        dir_in, 
        'derivatives',
        'fmriprep',
        f'sub-{sub}',
        'anat',
        f'sub-{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
    )
    path_out = os.path.join(
        dir_out, 
        'bet',
        f'sub-{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w_brain.nii.gz'
    )

    if not os.path.exists(path_in):
        print(f"No anatomical image for sub-{sub}")
        return 
   
    run_subprocess(['bet', path_in, path_out])


def process_sub_feat(sub: str, dir_in: str, dir_out: str, dir_fsl: str) -> None:

    # Set paths
    path_func = os.path.join(
        dir_in, 
        'derivatives',
        'fmriprep',
        f'sub-{sub}',
        'func',
        f'sub-{sub}_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold'
    )
    path_anat = os.path.join(
        dir_out, 
        'bet',
        f'sub-{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w_brain'
    )
    if not os.path.exists(path_func + '.nii.gz'):
        print(f"No functional image for sub-{sub}")
        return 
    if not os.path.exists(path_anat + '.nii.gz'):
        print(f"No anatomical image for sub-{sub}")
        return 

    # Create subject design file
    path_template = os.path.join('feat-designs', 'template.fsf')
    path_design = os.path.join('feat-designs', f'sub-{sub}.fsf')
    with open(path_template, 'r') as file:
        design = file.read()
    design = design.replace('DIR_OUT', os.path.join(dir_out, f'sub-{sub}'))
    design = design.replace('DIR_FSL', dir_fsl)
    design = design.replace('PATH_FUNC', path_func)
    design = design.replace('PATH_ANAT', path_anat)
    with open(path_design, 'w') as file:
        file.write(design)

    run_subprocess(['feat', path_design])
    

def process_sub_fix_extract(sub: str, dir_out: str) -> None:

    path = os.path.join(dir_out, f'sub-{sub}.feat')
    if not os.path.exists(path):
        print(f"No .feat directory for sub-{sub}")
        return
    
    run_subprocess(['fix', '-f', path])



def process_sub_fix_denoise(sub: str, dir_out: str, fix_model: str, threshold: int) -> None:
    
    path_sub = os.path.join(dir_out, f'sub-{sub}.feat')
    path_model = os.path.join(dir_out, f'{fix_model}.pyfix_model')
    if not os.path.exists(path_sub):
        print(f"No .feat directory for sub-{sub}")
        return

    run_subprocess(['fix', path_sub, path_model, str(threshold)])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str)  # group/datasets/ds002785
    parser.add_argument('--dir_out', type=str) # group/datasets/ds002785-fix
    parser.add_argument('--dir_fsl', type=str) # work/fsl
    parser.add_argument(
        '--step', type=str, 
        choices=['bet', 'feat', 'fix-extract', 'fix-train', 'fix-denoise']
    )
    parser.add_argument('--n_subs', type=int)
    parser.add_argument('--fix_model', type=str)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()
    
    os.makedirs(args.dir_out, exist_ok=True)
    if args.step == 'bet':
        os.makedirs(os.path.join(args.dir_out, 'bet'), exist_ok=True)
    
    step_fcns = {
        'bet': partial(process_sub_bet, dir_in=args.dir_in, dir_out=args.dir_out),
        'feat': partial(
            process_sub_feat, 
            dir_in=args.dir_in, 
            dir_out=args.dir_out, 
            dir_fsl=args.dir_fsl
        ),
        'fix-extract': partial(process_sub_fix_extract, dir_out=args.dir_out),
        'fix-denoise': partial(
            process_sub_fix_denoise, 
            dir_out=args.dir_out, 
            fix_model=args.fix_model, 
            threshold=args.threshold
        ),
    }

    # Subject labels
    subs = [str(s).zfill(4) for s in range(1, args.n_subs + 1)]
    
    # Train FIX model
    # NOTE: Only include subjects with hand labels in training
    if args.step == 'fix-train':
        path_model = os.path.join(args.dir_out, args.fix_model)
        paths_feat = [os.path.join(args.dir_out, f'sub-{sub}.feat') for sub in subs]
        run_subprocess(['fix', '-t', path_model, *paths_feat])
    
    # Execute other steps in parallel
    else:
        with mp.Pool(args.world_size) as pool:
            pool.map(step_fcns[args.step], subs)
