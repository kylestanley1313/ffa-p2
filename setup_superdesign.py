import os
import torch

from utils import gen_seeds, write_yaml


superdesign_name = 'sim-test-3'


# superdesign = { # sim-est-2-<r>

#     'n_reps': 75, #25,
#     'base_seed': 10002, # [10001, 10002, 10003]
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D', 'NetScheme2D'],
#     'n_facs': [[2, 2], [4, 4]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.05, 0.1],

#     'regime': [1, 2],
#     'n_sub': [5, 10, 20], #[2, 5, 10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }


# superdesign = { # sim-exp-2

#     'n_reps': 5, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.1],

#     'regime': [2],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],
#     'k_wise_sigmas': [None],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

# }

superdesign = { # sim-test-3

    'n_reps': 1, #25,
    'base_seed': 12345,
    'sz_space': [[30, 30]],

    'load_scheme': ['BumpScheme2D'],
    'n_facs': [[2, 2]], # (true, est)
    'err_scheme': ['GaussProc_Bump2D'],
    'delta': [0.1],

    'regime': [2],
    'n_sub': [20],
    'n_time': [500],

    'kernel_length_fac': [10], # TODO: How smooth is common component?
    # 'path_fac_cov': [
    #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
    #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
    # ],
    'kernel_length_err': [1], # TODO: How smooth is local component?
    
    'world_size_est': [2],
    'world_size_cov': [2],
    'n_folds_sub': [3],
    'n_folds_space_by_dim': [[2, 2]],
    'delta_est': [0.1],

    'k_wise_kappas': [None],
    'k_wise_sigmas': [None],

    # 'dssgd_bsz': [4096],
    # 'dssgd_lr': [64],
    # 'dssgd_tol': [1e-10],
    # 'dssgd_epochs': [10000],

}

# superdesign = { # sim-defense

#     'n_reps': 5, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.1],

#     'regime': [2],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_identity.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],
#     'k_wise_sigmas': [None],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

# }



# superdesign = { # sim-fse-2

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[40, 40]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.05, 0.1],

#     'regime': [1],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': [
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     # ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

#     'path_mask': [
#         '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_10-10.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_20-20.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_30-30.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_40-40.pt',           
#     ]

# }


###############################################################################
###############################################################################
###############################################################################


# superdesign = { # sim-test-2

#     'n_reps': 1, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05],

#     'regime': [1],
#     'n_sub': [2],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],

# }


# superdesign = { # sim-pilot-1

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D', 'NetScheme2D'],
#     'n_facs': [[2, 2], [4, 4]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.25, 0.5],
#     'n_sub': [10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }

# superdesign = { # sim-pilot-2, sim-pilot-4

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[20, 20], [40, 40]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.25],
#     'n_sub': [10],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#           '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_identity.pt',
#           '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt' # cor = 0.25
#         ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }

# superdesign = { # sim-pilot-3

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.05, 0.1, 0.15, 0.2],
#     'n_sub': [10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }

# superdesign = { # sim-pilot-5

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.1, 0.2, 0.3, 0.4],
#     'n_sub': [10, 20],
#     'n_time': [50],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }

# superdesign = { # sim-pilot-6

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.05],
#     'n_sub': [25, 50, 100, 200],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }

# superdesign = { # sim-pilot-7

#     'n_reps': 2, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[4, 4], [4, 8]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'prop_global': [0.25],
#     'n_sub': [10],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': None,

# }

# superdesign = { # sim-pilot-8

#     'n_reps': 2, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[4, 12], [4, 16]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'prop_global': [0.25],
#     'n_sub': [10],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': None,

# }

# superdesign = { # sim-pilot-9

#     'n_reps': 1, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[4, 12], [4, 16], [4, 20]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'prop_global': [0.1, 0.2],
#     'n_sub': [20],
#     'n_time': [50, 500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#         # '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],

# }


# superdesign = { # sim-pilot-10

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[10, 10], [20, 20], [40, 40], [80, 80]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     # 'prop_global': [0.2],
#     'regime': [1],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': [
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     # ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     # 'k_wise_kappas': None,

# }

# superdesign = { # sim-pilot-11

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'prop_global': [0.05, 0.1, 0.15, 0.2],
#     'n_sub': [10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'rand_scale': [0.2],


# }


# superdesign = { # sim-pilot-13

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D', 'NetScheme2D'],
#     'n_facs': [[2, 2], [4, 4]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.05, 0.1],

#     'regime': [1, 2],
#     'n_sub': [5, 10, 20], #[2, 5, 10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': ['/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_0.pt'],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

# }


# superdesign = { # sim-pilot-14

#     'n_reps': 2, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'regime': [1, 2],
#     'n_sub': [10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],

# }


# superdesign = { # sim-pilot-15

#     'n_reps': 2, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'regime': [1],
#     'n_sub': [10],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.15],

#     'k_wise_kappas': [None],

# }


# superdesign = { # sim-pilot-16

#     'n_reps': 2, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'regime': [1],
#     'n_sub': [10],
#     'n_time': [100],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],

# }

# superdesign = { # sim-ffa1-1

#     'n_reps': 1, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_BumpTensor2D'],
#     'delta': [0.1],

#     'regime': [1],
#     'n_sub': [10],
#     'n_time': [100],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         # '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],

# }


# superdesign = { # ffa2-bug-0

#     'n_reps': 1, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.1],

#     'regime': [2],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         # '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],
#     'k_wise_sigmas': [None],

# }

# superdesign = { # sim-test-0

#     'n_reps': 4, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.1],

#     'regime': [2],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         # '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],
#     'k_wise_sigmas': [None],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

# }

# superdesign = { # sim-exp-1

#     'n_reps': 4, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpTrioScheme2D'],
#     'n_facs': [[8, 8], [8, 25]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.1],

#     'regime': [2],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt',
#         '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt',
#     ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'k_wise_kappas': [None],
#     'k_wise_sigmas': [None],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

# }


# superdesign = { # sim-test-dssgd-1

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30]],

#     'load_scheme': ['BumpScheme2D', 'NetScheme2D'],
#     'n_facs': [[2, 2], [4, 4]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.05, 0.1],

#     'regime': [1, 2],
#     'n_sub': [5, 10, 20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     'path_fac_cov': [None],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     # 'k_wise_kappas': [None],
#     # 'k_wise_sigmas': [None],

#     # 'dssgd_bsz': [4096],
#     # 'dssgd_lr': [64],
#     # 'dssgd_tol': [1e-10],
#     # 'dssgd_epochs': [10000],

# }


# superdesign = { # sim-fse-0

#     'n_reps': 10, #25,
#     'base_seed': 12345,
#     'sz_space': [[30, 30], [60, 60], [90, 90]],

#     'load_scheme': ['BumpScheme2D'],
#     'n_facs': [[2, 2]], # (true, est)
#     'err_scheme': ['GaussProc_Bump2D'],
#     'delta': [0.05, 0.1],

#     'regime': [1],
#     'n_sub': [20],
#     'n_time': [500],

#     'kernel_length_fac': [10], # TODO: How smooth is common component?
#     # 'path_fac_cov': [
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt',
#     #     '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt',
#     # ],
#     'kernel_length_err': [1], # TODO: How smooth is local component?
    
#     'world_size_est': [2],
#     'world_size_cov': [2],
#     'n_folds_sub': [3],
#     'n_folds_space_by_dim': [[2, 2]],
#     'delta_est': [0.1],

#     'dssgd_bsz': [4096],
#     'dssgd_lr': [64],
#     'dssgd_tol': [1e-10],
#     'dssgd_epochs': [10000],

# }






abbs = {

    'BumpScheme2D': 'BL',
    'BumpScheme3D': 'BL',
    'NetScheme2D': 'NL',

    1: '1',
    2: '2',
    4: '4',
    5: '5',
    8: '8',
    10: '10',
    12: '12',
    16: '16',
    20: '20',
    25: '25',
    50: '50', 
    100: '100',
    200: '200',
    300: '300',
    400: '400',
    500: '500',

    'GaussProc_BumpTensor2D': 'BE',
    'GaussProc_BumpTensor3D': 'BE',
    'GaussProc_BSplinePinnedTensor2D': 'SE',

    0.05: '5',
    0.10: '10',
    0.15: '15',
    0.20: '20',
    0.30: '30',
    0.40: '40',

    (10, 10): '10',
    (20, 20): '20',
    (30, 30): '30',
    (40, 40): '40',
    (50, 50): '50',
    (60, 60): '60',
    (80, 80): '80',
    (90, 90): '90',

    # Factor covariances
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_identity.pt': 'orth',
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_1.pt': 'obl1',
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-2_2.pt': 'obl2',

    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_identity.pt': 'orth',
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_1.pt': 'obl1',
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-4_2.pt': 'obl2',

    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_identity.pt': 'orth',
    '/storage/home/kms8227/work/ffa-p2-priv/fac-covs/cov_k-8_1.pt': 'obl',

    # Masks
    '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_10-10.pt': '10',
    '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_20-20.pt': '20',
    '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_30-30.pt': '30',
    '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_40-40.pt': '40', 

}


# Study: Subspace Estimation
gen = torch.Generator().manual_seed(superdesign['base_seed'])
for load_scheme in superdesign['load_scheme']:
    for regime in superdesign['regime']:
        for n_facs in superdesign['n_facs']:

            design = superdesign.copy()
            design['load_scheme'] = [load_scheme]
            design['regime'] = [regime]
            design['n_facs'] = [n_facs]
            design['base_seed'] = gen_seeds(gen, 1)

            design_name = (f"{superdesign_name}_regime-{abbs[regime]}_loads-{abbs[load_scheme]}_K-{abbs[n_facs[0]]}")
            print(design_name)
            path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
            write_yaml(design, path)
# gen = torch.Generator().manual_seed(superdesign['base_seed'])
# for n_sub in superdesign['n_sub']:
#     for delta in superdesign['delta']:

#         design = superdesign.copy()
#         design['n_sub'] = [n_sub]
#         design['delta'] = [delta]
#         design['base_seed'] = gen_seeds(gen, 1)

#         design_name = (f"{superdesign_name}_nsub-{abbs[n_sub]}_delta-{abbs[delta]}")
#         print(design_name)
#         path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
#         write_yaml(design, path)
# gen = torch.Generator().manual_seed(superdesign['base_seed'])
# for prop_glob in superdesign['prop_global']:
#     for delta in superdesign['delta']:

#         design = superdesign.copy()
#         design['prop_glob'] = [prop_glob]
#         design['delta'] = [delta]
#         design['base_seed'] = gen_seeds(gen, 1)

#         design_name = (f"{superdesign_name}_pg-{abbs[prop_glob]}_delta-{abbs[delta]}")
#         print(design_name)
#         path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
#         write_yaml(design, path)
# gen = torch.Generator().manual_seed(superdesign['base_seed'])
# for regime in superdesign['regime']:
#     for n_facs in superdesign['n_facs']:

#         design = superdesign.copy()
#         design['regime'] = [regime]
#         design['n_facs'] = [n_facs]
#         design['base_seed'] = gen_seeds(gen, 1)

#         design_name = (f"{superdesign_name}_regime-{abbs[regime]}_{abbs[n_facs[1]]}")
#         print(design_name)
#         path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
#         write_yaml(design, path)


# Study: Subspace Expression
# gen = torch.Generator().manual_seed(superdesign['base_seed'])
# for n_facs in superdesign['n_facs']:
#     for path_fac_cov in superdesign['path_fac_cov']:
#         design = superdesign.copy()
#         design['n_facs'] = [n_facs]
#         design['path_fac_cov'] = [path_fac_cov]
#         design['base_seed'] = gen_seeds(gen, 1)

#         design_name = (f"{superdesign_name}_ktrue-{abbs[n_facs[0]]}_kest-{abbs[n_facs[1]]}_faccov-{abbs[path_fac_cov]}")
#         print(design_name)
#         path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
#         write_yaml(design, path)



        
# Study: Factor Score Estimation
# gen = torch.Generator().manual_seed(superdesign['base_seed'])
# for path_mask in superdesign['path_mask']:
#     for delta in superdesign['delta']:
#             # for path_fac_cov in superdesign['path_fac_cov']:

#             design = superdesign.copy()
#             design['path_mask'] = [path_mask]
#             design['delta'] = [delta]
#             # design['path_fac_cov'] = [path_fac_cov]
#             design['base_seed'] = gen_seeds(gen, 1)

#             design_name = (f"{superdesign_name}_mask-{abbs[path_mask]}_delta-{abbs[delta]}")
#             print(design_name)
#             path = os.path.join('/storage/group/kms8227/default/ffa-p2-priv/designs', f'{design_name}.yml')
#             # os.remove(path)
#             write_yaml(design, path)






