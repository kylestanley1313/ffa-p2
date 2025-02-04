import argparse
import nibabel as nib
import os
import torch

from config import load_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--n_subs', type=int)
    parser.add_argument('--mask_name', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    mask = torch.ones(65, 77, 60).to(bool)
    subs = [str(s).zfill(4) for s in range(1, args.n_subs + 1)]
    for sub in subs:
        print(sub)
        
        path = os.path.join(
            config.group_dir,
            'datasets/ds002785/derivatives/fmriprep',
            f'sub-{sub}/func',
            f'sub-{sub}_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        )

        try: 
            mask_ = nib.load(path).get_fdata()
            mask_ = torch.from_numpy(mask_).to(bool)
            mask = torch.logical_and(mask, mask_)
        except FileNotFoundError:
            print(f"Mask image not found for sub-{sub}.")
    
    print(f"Number of Brain Voxels: {torch.sum(mask).item()}")
    

    # Save mask
    path = os.path.join('masks', f'{args.mask_name}.pt')
    torch.save(mask, path)

    # NOTE: There are 53,784 brain voxels in allsub mask.





###############################################################################
###############################################################################
###############################################################################

# NOTE: The commented out code below creates a mask by ignoring "bad" voxels.
# This code can likely be deleted later. 

# import argparse
# import os
# import torch

# from config import load_config
# from utils import get_sub_path, read_sub_file


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str)
#     parser.add_argument('--path_mask_in', type=str)
#     parser.add_argument('--path_mask_out', type=str)
#     parser.add_argument('--z_min', type=int, required=False)
#     parser.add_argument('--z_max', type=int, required=False)
#     parser.add_argument('--all_subs', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)
#     mask = torch.load(args.path_mask_in)
#     n_voxels_start = torch.sum(mask)

#     # ---------- Slice Exclusion ---------- #
#     if args.z_min is not None and args.z_max is not None: 
#         z_idx = torch.arange(mask.shape[2])
#         z_mask = torch.logical_or(z_idx < args.z_min, z_idx > args.z_max)
#         mask[:,:,z_mask] = False

#     # ---------- Voxel Exclusion ---------- #
#     # NOTE: Excludes any voxel having a time course (across all subjects) that
#     # is identically zero at any point. 
#     bad_voxels = torch.empty(0, 3, dtype=torch.int32)
#     if args.all_subs: 
            
#             for n in range(216):
            
#                 # Check if file exists
#                 path = get_sub_path(n)
#                 if not os.path.exists(path):
#                     print(f"Subject {n} not found!")
#                     continue

#                 # Read, mask, then extract bad voxels from scan
#                 scan = read_sub_file(n)
#                 scan[~mask,:] = -1
#                 has_zero = (scan == 0).any(dim=-1)
#                 voxels = torch.nonzero(has_zero, as_tuple=False)

#                 # Update bad_voxels
#                 bad_voxels = torch.cat((bad_voxels, voxels), dim=0)
#                 bad_voxels = torch.unique(bad_voxels, dim=0)

#             # Update mask
#             mask_ = torch.ones(65, 77, 60).to(bool)
#             mask_[bad_voxels[:, 0], bad_voxels[:, 1], bad_voxels[:, 2]] = False
#             mask = torch.logical_and(mask, mask_)


#     else: 
#         raise Exception('Invalid command line arguments!')
    
#     # Save mask
#     n_voxels_end = torch.sum(mask)
#     print(f"Voxels Removed: {n_voxels_start - n_voxels_end} of {n_voxels_start}")
#     torch.save(mask, args.path_mask_out)

