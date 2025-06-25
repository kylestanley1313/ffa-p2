import argparse
import nibabel as nib
import numpy as np
import os
import torch
import torch.multiprocessing as mp

from config import load_config


SZ_SPACE = [65, 77, 60]
N_VARS = 300300
N_TIME = 470



def bandpass_filter(time_series, tr=0.75, min_hz=0.01, max_hz=0.15):
    """
    Bandpass filters a time series to retain frequencies between min_hz and max_hz.
    
    Args:
        time_series (torch.Tensor): 1D tensor containing the time series data.
        tr (float): Time between observations (in seconds).
        min_hz (float): Minimum frequency to retain (in Hz).
        max_hz (float): Maximum frequency to retain (in Hz).
    
    Returns:
        torch.Tensor: Bandpass-filtered time series.
    """
    n = time_series.size(0)
    # Sampling frequency
    fs = 1 / tr

    # Compute the Fourier transform
    fft_values = torch.fft.fft(time_series)
    
    # Compute the corresponding frequencies
    freqs = torch.fft.fftfreq(n, d=tr)
    
    # Create a mask for the desired frequency range
    bandpass_mask = (freqs >= min_hz) & (freqs <= max_hz) | (freqs <= -min_hz) & (freqs >= -max_hz)
    
    # Apply the mask to the Fourier coefficients
    filtered_fft_values = fft_values * bandpass_mask
    
    # Perform the inverse Fourier transform
    filtered_time_series = torch.fft.ifft(filtered_fft_values).real

    return filtered_time_series


def prepare_subject(args) -> None:

    # Unpack args
    sub_num, sub_cnt, path_mask, dir_in, dir_out = args

    # Load mask
    mask = torch.flatten(torch.load(path_mask))
    # nz_idx = torch.nonzero(mask)

    # Load FIXed AOMIC data
    sub_lab = str(sub_num).zfill(4)
    path = os.path.join(
        dir_in,
        f'sub-{sub_lab}.feat/filtered_func_data_clean.nii.gz'
    )
    data = nib.load(path).get_fdata()
    data = torch.from_numpy(data).to(torch.float32) 

    # Apply bandpass filter
    # data = data.reshape(N_VARS, N_TIME)
    # for idx in nz_idx: 
    #     v = idx.item()
    #     data[v] = torch.tensor(bandpass_filter(data[v]))
    # data = data.reshape(*SZ_SPACE, N_TIME)

    # Apply variance normalization
    mean = data.mean(dim=-1, keepdim=True)
    data = data - mean
    std = data.std(dim=-1, keepdim=True)
    data = data / std

    # Replace NaNs with zeros
    data = torch.nan_to_num(data, nan=0.0)

    # Permute dimensions then write
    data = data.permute(3, 0, 1, 2)
    path = os.path.join(dir_out, f'data_n-{sub_cnt}_i-0_.pt')
    torch.save(data, path)

    print(f"sub-{sub_lab} complete", flush=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--analysis', type=str)
    parser.add_argument('--n_subs', type=int)
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    dir_in = os.path.join(config.group_dir, 'datasets', 'ds002785-fix') # contains FIX-cleaned data
    dir_out = os.path.join(config.group_root, 'datasets', args.analysis) # to contain FFA-prepared data

    subs = [str(s).zfill(4) for s in range(1, args.n_subs + 1)]
    # subs = ['0016', '0017', '0018'] ###
    sub_num = 1 ###
    sub_cnt = 0
    args_list = []
    for sub in subs: 
        path = os.path.join(
            dir_in,
            f'sub-{sub}.feat',
            'filtered_func_data_clean.nii.gz'
        )
        if os.path.exists(path):
            args_list.append(
                (sub_num, sub_cnt, args.path_mask, dir_in, dir_out)
            )
            sub_cnt += 1
        else: 
            print(f"WARNING: Path not found for subject {sub_num}.", flush=True)
        sub_num += 1

    # Prepare in parallel
    with mp.Pool(args.world_size) as pool:
        pool.map(prepare_subject, args_list)


