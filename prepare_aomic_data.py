import argparse
import nibabel as nib
import os
import torch
import torch.multiprocessing as mp

from config import load_config



def prepare_subject(args) -> None:

    # Unpack args
    sub_num, sub_cnt, dir_in, dir_out = args

    # Load FIXed AOMIC data
    sub_lab = str(sub_num).zfill(4)
    path = os.path.join(
        dir_in,
        f'sub-{sub_lab}.feat/filtered_func_data_clean.nii.gz'
    )
    data = nib.load(path).get_fdata()
    data = torch.from_numpy(data).to(torch.float32) 

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--analysis', type=str)
    parser.add_argument('--n_subs', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    dir_in = os.path.join(config.group_dir, 'datasets', 'ds002785-fix') # contains FIX-cleaned data
    dir_out = os.path.join(config.group_root, 'datasets', args.analysis) # to contain FFA-prepared data

    subs = [str(s).zfill(4) for s in range(1, args.n_subs + 1)]
    sub_num = 1
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
                (sub_num, sub_cnt, dir_in, dir_out)
            )
            sub_cnt += 1
        else: 
            print(f"WARNING: Path not found for subject {sub_num}.")
        sub_num += 1

    # Prepare in parallel
    with mp.Pool(args.world_size) as pool:
        pool.map(prepare_subject, args_list)



###############################################################################
###############################################################################
###############################################################################

# NOTE: Commented out code below prepares AOMIC data via other means 
# (e.g., prewhitening) that are no longer used. It may be deleted later. 


# SZ_SPACE = [65, 77, 60]
# N_VARS = 300300
# N_TIME = 480
# BURN_IN = 10


# Fine tune these whitening params
# def whiten_wavelet(series):

#     # Wavelet decomposition
#     wavelet = 'db4'  # Daubechies wavelet
#     level = pywt.dwt_max_level(
#         data_len=len(series), 
#         filter_len=pywt.Wavelet(wavelet).dec_len
#     )
#     coeffs = pywt.wavedec(series, wavelet, level=level)

#     # Thresholding (for noise removal)
#     threshold = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimation using robust MAD
#     coeffs_thresholded = [
#         pywt.threshold(c, threshold, mode='soft') for c in coeffs
#     ]

#     # Reconstruct the signal from thresholded coefficients
#     prewhitened_time_course = pywt.waverec(coeffs_thresholded, wavelet)

#     # return prewhitened_time_course
#     return zscore(series - prewhitened_time_course)


# # TODO: Fine tune whitening params
# def whiten_arima(series):
#     model = ARIMA(series, order=(2, 1, 2), enforce_stationarity=False)
#     fit = model.fit()
#     return zscore(fit.resid)


# def bandpass_filter(time_series, tr=0.75, min_hz=0.01, max_hz=0.1):
#     """
#     Bandpass filters a time series to retain frequencies between min_hz and max_hz.
    
#     Args:
#         time_series (torch.Tensor): 1D tensor containing the time series data.
#         tr (float): Time between observations (in seconds).
#         min_hz (float): Minimum frequency to retain (in Hz).
#         max_hz (float): Maximum frequency to retain (in Hz).
    
#     Returns:
#         torch.Tensor: Bandpass-filtered time series.
#     """
#     n = time_series.size(0)
#     # Sampling frequency
#     fs = 1 / tr

#     # Compute the Fourier transform
#     fft_values = torch.fft.fft(time_series)
    
#     # Compute the corresponding frequencies
#     freqs = torch.fft.fftfreq(n, d=tr)
    
#     # Create a mask for the desired frequency range
#     bandpass_mask = (freqs >= min_hz) & (freqs <= max_hz) | (freqs <= -min_hz) & (freqs >= -max_hz)
    
#     # Apply the mask to the Fourier coefficients
#     filtered_fft_values = fft_values * bandpass_mask
    
#     # Perform the inverse Fourier transform
#     filtered_time_series = torch.fft.ifft(filtered_fft_values).real

#     return filtered_time_series


# def prepare_subject_1(args) -> None:

#     # Unpack args
#     sub_num, sub_cnt, path_mask, dir_dataset = args

#     # Load mask
#     mask = torch.flatten(torch.load(path_mask))
#     nz_idx = torch.nonzero(mask)

#     # Load AOMIC data
#     data = read_sub_file(sub_num)
#     data = data.reshape(N_VARS, N_TIME - BURN_IN)

#     # Whiten brain voxel time courses
#     cnt = 0
#     for idx in nz_idx: 
#         v = idx.item()
#         series = data[v].numpy()

#         # Whiten time course
#         if np.any(series != 0):
#             # data[v] = torch.tensor(whiten_wavelet(series))
#             data[v] = bandpass_filter(torch.tensor(series))
        
#         cnt += 1
#         if cnt % 10000 == 0: 
#             print(f"sub_num = {sub_num} | cnt = {cnt}", flush=True)

#     # Reshape and save data
#     data = data.reshape(*SZ_SPACE, N_TIME - BURN_IN)
#     data = data.permute(3, 0, 1, 2)
#     path = os.path.join(dir_dataset, f'data_n-{sub_cnt}_i-0_.pt') # TODO: WRONG NAMING CONVENTION!
#     torch.save(data, path)


    # ---------- Non-FIX Preparation ---------- #

    # # Get list of subject files
    # # TODO: Handle num_subs too large
    # sub_num = 1
    # sub_cnt = 0
    # args_list = []
    # while len(args_list) < args.num_subs:
    #     sub_path = get_sub_path(sub_num)
    #     if os.path.exists(sub_path):
    #         args_list.append(
    #             (sub_num, sub_cnt, args.path_mask, dir_dataset)
    #         )
    #         sub_cnt += 1
    #     else: 
    #         print(f"WARNING: Path not found for subject {sub_num}.")
    #     sub_num += 1

    # # Whiten in parallel
    # with mp.Pool(args.world_size) as pool:
    #     pool.map(prepare_subject_1, args_list)











    # # Compute mean of AOMIC data
    # print(f"# ========== Computing Mean Scan ========== #")
    # cnt = 0
    # mean_scan = 0
    # for i in range(1, args.num_subs + 1):
    #     print(f"{i} of {args.num_subs}")

    #     # Read in raw scan
    #     try: 
    #         scan = read_sub_file(i)
    #     except FileNotFoundError:
    #         print(f"WARNING: File not found.")
    #         continue

    #     # Update mean scan
    #     scan -= scan.mean(dim=-1, keepdim=True)
    #     mean_scan += scan

    #     cnt += 1

    # scan /= cnt

    # # Write centered scans to /datasets
    # print(f"# ========== Centering, Scaling, and Writing Scans ========== #")
    # dir_dataset = os.path.join(config.scratch_root, 'datasets', args.analysis)
    # if not os.path.exists(dir_dataset):
    #     os.mkdir(dir_dataset)
    # cnt = 0
    # for i in range(1, args.num_subs + 1):
    #     print(f"{i} of {args.num_subs}")

    #     # Read in raw scan
    #     try: 
    #         scan = read_sub_file(i)
    #     except FileNotFoundError:
    #         print(f"WARNING: File not found.")
    #         continue    
    
    #     # Center, scale, permute, and save scan
    #     scan -= scan.mean(dim=-1, keepdim=True) - mean_scan
    #     scan /= 1000
    #     scan = scan.permute(3, 0, 1, 2)
    #     path = os.path.join(dir_dataset, f'data_i-{cnt}_.pt')
    #     torch.save(scan, path)

    #     cnt += 1











    # # Compute mean of AOMIC data
    # print(f"# ========== Computing Mean Scan ========== #")
    # cnt = 0
    # mean_scan = 0
    # for i in range(1, args.num_subs + 1):
    #     print(f"{i} of {args.num_subs}")

    #     # Read in raw scan
    #     try: 
    #         scan = read_sub_file(i)
    #     except FileNotFoundError:
    #         print(f"WARNING: File not found.")
    #         continue

    #     # Update mean scan
    #     mean_scan += scan

    #     cnt += 1

    # scan /= cnt

    # # Write centered scans to /datasets
    # print(f"# ========== Centering, Scaling, and Writing Scans ========== #")
    # dir_dataset = os.path.join(config.scratch_root, 'datasets', args.analysis)
    # if not os.path.exists(dir_dataset):
    #     os.mkdir(dir_dataset)
    # cnt = 0
    # for i in range(1, args.num_subs + 1):
    #     print(f"{i} of {args.num_subs}")

    #     # Read in raw scan
    #     try: 
    #         scan = read_sub_file(i)
    #     except FileNotFoundError:
    #         print(f"WARNING: File not found.")
    #         continue    
    
    #     # Center, scale, permute, and save scan
    #     scan -= mean_scan
    #     scan /= 1000
    #     scan = scan.permute(3, 0, 1, 2)
    #     path = os.path.join(dir_dataset, f'data_i-{cnt}_.pt')
    #     torch.save(scan, path)

    #     cnt += 1














    # # Prepare each scan
    # print(f"# ========== Preparing Scans ========== #")
    # dir_dataset = os.path.join(config.scratch_root, 'datasets', args.analysis)
    # cnt = 0
    # for i in range(1, args.num_subs + 1):
    #     print(f"{i} of {args.num_subs}")

    #     # Read in raw scan
    #     try: 
    #         scan = read_sub_file(i)
    #     except FileNotFoundError:
    #         print(f"WARNING: File not found.")
    #         continue

    #     # Apply variance normalization
    #     mean = scan.mean(dim=-1, keepdim=True)
    #     scan = scan - mean
    #     # std = scan.std(dim=-1, keepdim=True)
    #     # scan = scan / std
    #     scan = scan / 1000

    #     # Replace NaNs with zeros
    #     scan = torch.nan_to_num(scan, nan=0.0)

    #     # Permute dimensions then write
    #     scan = scan.permute(3, 0, 1, 2)
    #     path = os.path.join(dir_dataset, f'data_i-{cnt}_.pt')
    #     torch.save(scan, path)

    #     cnt += 1

    

