import argparse
import os
import torch

from utils import load_config, model_from_loads

from typing import List



def shrink_loads(loads: torch.Tensor, kappas: List[float]) -> torch.Tensor:
    """Shrink loadings via:
        sgn(l_{km}) * max(0, |l_{km}| - kappas_k / |l_{km}|^2)
    """
    loads_abs = torch.abs(loads)
    loads_sgn = torch.sign(loads)
    kappas = torch.tensor(kappas, dtype=torch.float32).view(-1, 1)
    shrink_term = kappas / torch.where(loads_abs**2 != 0, loads_abs**2, torch.ones_like(loads_abs))
    return loads_sgn * torch.clamp(loads_abs - shrink_term, min=0.0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--kappas', type=float, nargs='+')
    parser.add_argument('--path_kappas', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--est_method', type=str)
    parser.add_argument('--rot_method', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.path_kappas is not None: 
        kappas = torch.load(args.path_kappas)
    elif args.kappas is not None: 
        kappas = args.kappas
    else: 
        raise Exception("Must pass kappas or kappa path!")

    # Path preparation
    if args.split == 'full': 
        fname_in = f'model-{args.est_method}-{args.split}-{args.rot_method}-smooth.pth'
        fname_out = f'model-{args.est_method}-{args.split}-{args.rot_method}-smooth-shrink.pth'
    else: 
        fname_in = f'model-{args.est_method}-{args.split}-{args.fold}-{args.rot_method}-smooth.pth'
        fname_out = f'model-{args.est_method}-{args.split}-{args.fold}-{args.rot_method}-smooth-shrink.pth'
    path_in = os.path.join(args.dir_out, fname_in)
    path_out = os.path.join(args.dir_out, fname_out)


    # Read in loadings
    state_dict = torch.load(path_in)
    loads = state_dict['loads'].t()

    if len(kappas) != loads.shape[0]: 
        raise Exception(f"Must pass {loads.shape[0]} kappas!")
    
    # Shrink loadings
    shrunk = shrink_loads(loads, kappas)

    # Save shrunk model
    model = model_from_loads(shrunk.t())
    torch.save(model.state_dict(), path_out)
    