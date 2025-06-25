import argparse
import os
import torch
import torch.nn.functional as fcnl

from config import load_config
from utils import multiply_list, model_from_loads

# NOTE: This script implements smoothing via convolution with a Gaussian
# kernel. To accelerate smoothing, D separable 1-dimensional convolutions are
# applied along each dimension. If the data is masked, missing values are 
# ignored during smoothing. 

def gaussian_kernel_1d(size: int, sigma: float):
    """Creates a 1D Gaussian kernel."""
    x = torch.arange(size) - size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()  # normalize to sum to 1
    return kernel.view(1, 1, -1)  # shape (1, 1, size) for conv1d


def gaussian_filter(loads: torch.Tensor, sigma: float):
    """Applies a Gaussian filter to a tensor while preserving NaN locations."""
    
    if sigma == 0: 
        
        return loads
    
    else: 
    
        nan_mask = torch.isnan(loads)  # save original NaN locations
        mask = (~nan_mask).float()  # 1 where valid, 0 where NaN
        loads = torch.nan_to_num(loads, nan=0.0)  # replace NaNs with 0
        
        dim = loads.ndimension()
        smoothed = loads.clone()
        smoothed_mask = mask.clone()

        for d in range(dim):
            kernel_size = int(6 * sigma) | 1  # Ensure odd size
            kernel = gaussian_kernel_1d(kernel_size, sigma).to(loads.device)

            if dim == 1:
                smoothed = fcnl.conv1d(smoothed.unsqueeze(0), kernel, padding=kernel_size // 2).squeeze(0)
                smoothed_mask = fcnl.conv1d(smoothed_mask.unsqueeze(0), kernel, padding=kernel_size // 2).squeeze(0)
            elif dim == 2:
                smoothed = smoothed.unsqueeze(0).unsqueeze(0)
                smoothed_mask = smoothed_mask.unsqueeze(0).unsqueeze(0)
                smoothed = fcnl.conv2d(smoothed, kernel.view(1, 1, kernel_size, 1), padding=(kernel_size // 2, 0))
                smoothed = fcnl.conv2d(smoothed, kernel.view(1, 1, 1, kernel_size), padding=(0, kernel_size // 2))
                smoothed_mask = fcnl.conv2d(smoothed_mask, kernel.view(1, 1, kernel_size, 1), padding=(kernel_size // 2, 0))
                smoothed_mask = fcnl.conv2d(smoothed_mask, kernel.view(1, 1, 1, kernel_size), padding=(0, kernel_size // 2))
                smoothed = smoothed.squeeze(0).squeeze(0)
                smoothed_mask = smoothed_mask.squeeze(0).squeeze(0)
            elif dim == 3:
                smoothed = smoothed.unsqueeze(0).unsqueeze(0)
                smoothed_mask = smoothed_mask.unsqueeze(0).unsqueeze(0)
                smoothed = fcnl.conv3d(smoothed, kernel.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))
                smoothed = fcnl.conv3d(smoothed, kernel.view(1, 1, 1, kernel_size, 1), padding=(0, kernel_size // 2, 0))
                smoothed = fcnl.conv3d(smoothed, kernel.view(1, 1, 1, 1, kernel_size), padding=(0, 0, kernel_size // 2))
                smoothed_mask = fcnl.conv3d(smoothed_mask, kernel.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))
                smoothed_mask = fcnl.conv3d(smoothed_mask, kernel.view(1, 1, 1, kernel_size, 1), padding=(0, kernel_size // 2, 0))
                smoothed_mask = fcnl.conv3d(smoothed_mask, kernel.view(1, 1, 1, 1, kernel_size), padding=(0, 0, kernel_size // 2))
                smoothed = smoothed.squeeze(0).squeeze(0)
                smoothed_mask = smoothed_mask.squeeze(0).squeeze(0)

        # Avoid division by zero in areas with no data
        smoothed /= torch.clamp(smoothed_mask, min=1e-6)

        # Restore NaN values in their original locations
        smoothed[nan_mask] = float('nan')

        return smoothed



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--sigmas', type=float, nargs='+')
    parser.add_argument('--path_sigmas', type=str)
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument('--split', type=str, choices=['full', 'train'])
    parser.add_argument('--fold', type=int)
    parser.add_argument('--est_method', type=str)
    parser.add_argument('--rot_method', type=str)
    parser.add_argument('--path_mask', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.path_sigmas is not None: 
        sigmas = torch.load(args.path_sigmas)
    elif args.sigmas is not None: 
        sigmas = args.sigmas
    else: 
        raise Exception("Must pass sigmas or sigma path!")
    
    # Read in loadings
    if args.split == 'train':
        fname_model = f'model-{args.est_method}-train-{args.fold}-{args.rot_method}'
    else: 
        fname_model = f'model-{args.est_method}-full-{args.rot_method}'
    path = os.path.join(args.dir_out, f'{fname_model}.pth')
    state_dict = torch.load(path)
    loads = state_dict['loads'].t()
    n_facs = loads.shape[0]

    if args.path_mask is not None: 

        # Read in mask
        mask = torch.load(args.path_mask)
        # sz_space = mask.shape
        mask_flat = torch.flatten(mask)
        n_voxels = len(mask_flat)

        # Create masked loadings
        loads_masked = torch.full((n_facs, n_voxels), float('nan'))
        loads_masked[:,mask_flat] = loads
        loads_masked = loads_masked.reshape((n_facs, *args.sz_space))
    
    else: 
        # sz_space = loads.shape[1:]
        n_voxels = multiply_list(args.sz_space)
        mask_flat = torch.ones(n_voxels).to(bool)
        loads_masked = loads.reshape((n_facs, *args.sz_space))

    # Smooth masked loadings
    smoothed = torch.zeros((n_facs, *args.sz_space))
    for k in range(n_facs):
        smoothed[k] = gaussian_filter(loads_masked[k], sigmas[k])

    # Create smooth model
    smoothed = smoothed.reshape(n_facs, n_voxels)[:,mask_flat]
    model = model_from_loads(smoothed.t())
    
    # Save smooth model
    path = os.path.join(args.dir_out, f'{fname_model}-smooth.pth')
    torch.save(model.state_dict(), path)


