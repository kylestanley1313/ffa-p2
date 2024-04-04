import argparse
import numpy as np
import os
import torch

from utils.utils import compute_loss, model_from_loads


KAPPAS = torch.arange(0, 1, step=0.001).tolist()


def shrink_loads(loads: torch.Tensor, kappa: float) -> torch.Tensor:
    num_facs = loads.shape[0]
    for k in range(num_facs):
        temp = np.abs(loads[k]) - kappa/np.abs(loads[k])**2
        temp[temp < 0] = 0
        loads[k] = np.sign(loads[k]) * temp
    return loads
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    args = parser.parse_args()

    # Set paths
    dir_cov = os.path.join(args.dir_out, 'cov')
    path_kappa = os.path.join(args.dir_out, 'kappa.pt')
    path_in_train = os.path.join(args.dir_out, f'model-train-rot.pth')
    path_in_full = os.path.join(args.dir_out, f'model-full-rot.pth')
    path_out_full = os.path.join(args.dir_out, f'model-full-pp.pth')

    # Optimally shrink the roated loadings
    loads_rot = torch.load(path_in_train)['loads']
    best_valid_loss = float('inf')
    best_kappa = None
    for kappa in KAPPAS:

        # Shrink loadings then comptue validation loss
        loads_pp = shrink_loads(loads_rot, kappa)
        model_pp = model_from_loads(loads_pp)
        valid_loss = compute_loss(model_pp, dir_cov, 'valid')

        print(f"kappa = {kappa} | valid_loss = {valid_loss}")

        # Break from loop if improvement stops or update best kappa
        if valid_loss > best_valid_loss:
            break
        best_valid_loss = valid_loss
        best_kappa = kappa

    # Shrink the full rotated model using the best kappa
    torch.save(torch.tensor(best_kappa, dtype=torch.float64), path_kappa)
    loads_rot = torch.load(path_in_full)['loads']
    loads_pp = shrink_loads(loads_rot, best_kappa)
    model_pp = model_from_loads(loads_pp)
    torch.save(model_pp, path_out_full)

