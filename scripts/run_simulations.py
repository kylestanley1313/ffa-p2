import argparse
import os
import torch

from utils import (
    execute_script, 
    load_config,
    load_yaml, 
    multiply_list,
)


def get_all_rep_paths(config, design_id): 
    paths = []
    dir_design = os.path.join(config.group_root, 'designs', design_id)
    for root, _, files in os.walk(dir_design):
        for file in files:
            paths.append(os.path.join(root, file))
    return sorted(paths)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument(
        '--steps', type=str, nargs='+',
        choices=[

            'setup-simulations',
            'simulate-data',
            'compute-covariance',
            'allocate-points',
            'initialize-loadings',
            'estimate-loadings',
            'rotate',
            'tune-sigmas',
            'smooth-loadings',
            'tune-kappas',
            'shrink-loadings',
            'tune-gammas',
            'estimate-factor-scores',

            'melodic-data-prep',
            'melodic-tune-sigma',
            'melodic-estimation',

        ]
    )
    parser.add_argument('--all_le_steps', action='store_true')
    parser.add_argument('--all_fse_steps', action='store_true')
    parser.add_argument(
        '--methods_le', type=str, nargs='+',
        choices=['lbfgs', 'dsgd', 'dssgd'],
        default=['lbfgs', 'dsgd', 'dssgd']
    )
    parser.add_argument(
        '--methods_fse', type=str, nargs='+', 
        choices=['pls', 'pgls', 'rbels', 'rbegls']
    )
    parser.add_argument(
        '--rotations', type=str, nargs='+',
        choices=['varimax', 'quartimax', 'quartimin'],
        default=['varimax', 'quartimin']
    )
    parser.add_argument(
        '--regimes_fse', type=int, nargs='+',
        choices=[1, 2, 3]
    )
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--silent_fail', action='store_true')
    args = parser.parse_args()

    # Validate command-line arguments
    if args.methods_fse:
        assert args.regimes_fse is not None, "Must pass FSE regimes!"
        assert len(args.methods_le) == 1, "Must pass only one LE method!"

    config = load_config(args.config)
    raise_error = not args.silent_fail

    # Set steps. The `all_steps` flags override `steps` flag.
    if args.all_le_steps: 
        steps = [
            'setup-simulations',
            'simulate-data',
            'compute-covariance',
            'allocate-points',
            'initialize-loadings',
            'tune-alpha',
            'estimate-loadings',
        ]
    elif args.all_fse_steps:
        steps = [
            'setup-simulations',
            'simulate-data',
            'compute-covariance',
            'allocate-points',
            'initialize-loadings',
            'tune-alpha',
            'estimate-loadings',
            # 'compute-inv-err-cov',
            'tune-gammas',
            'estimate-factor-scores',
        ]
    else: 
        steps = args.steps


    # Setup simulation files
    if 'setup-simulations' in steps:
        path = os.path.join(config.root, 'scripts', 'setup_simulations.py')
        flags = {
            'config': args.config,
            'design': args.design,
        }
        execute_script(path, flags, raise_error)


    # Get all repetitions
    reps = []
    for path in get_all_rep_paths(config, args.design):
        rep = load_yaml(path)

        # Set n_vars
        path_mask = rep.get('path_mask')
        if path_mask is None: 
            rep['n_vars'] = multiply_list([int(d) for d in rep['sz_space']])
        else: 
            rep['n_vars'] = torch.nonzero(torch.load(path_mask)).shape[0]

        rep['id'] = (path.split('designs/')[-1]
                     .replace('.yml', '')
                     .replace('/', '_'))
        reps.append(rep)


    if 'simulate-data' in steps:
        print(f"\n{'='*40} DATA SIMULATION {'='*40}\n")
        for rep in reps: 
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', 'simulate_data.py')
            flags = {
                'config': args.config,
                'dir_dataset': rep['dir_dataset'],
                'dir_dataset_truth': rep['dir_dataset_truth'],
                'dir_out': rep['dir_out'],
                'n_sub': rep['n_sub'],
                'n_time': rep['n_time'],
                'sz_space': rep['sz_space'],
                'load_scheme': rep['load_scheme'],
                'err_scheme': rep['err_scheme'],
                'n_facs': rep['n_facs'][0],
                'delta': rep['delta'],
                # 'prop_global': rep['prop_global'],
                'regime': rep['regime'],
                'batch_size': 1000,
                'kernel_length_fac': rep['kernel_length_fac'],
                # 'kernel_length_fac_var': rep['kernel_length_fac_var'],
                'kernel_length_err': rep['kernel_length_err'],
                # 'kernel_length_err_var': rep['kernel_length_err_var'],
                'seed': rep['seed'],
            }
            if rep.get('path_fac_cov'):
                flags['path_fac_cov'] = rep['path_fac_cov']
            if not rep['id'].endswith('-0'):  # Read true loads and errs
                flags['read_loads'] = None
                # flags['read_errs'] = None # TODO: Implement subject-wise errors?
            execute_script(path, flags, raise_error)


    if 'compute-covariance' in steps:
        print(f"\n{'='*40} COVARIANCE COMPUTATION {'='*40}\n")
        for rep in reps: 
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', 'compute_covariance.py')
            flags = {
                'config': args.config,
                'dir_dataset': rep['dir_dataset'],
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'world_size': rep['world_size_cov'],
                'delta': rep['delta_est'],
                'n_folds_sub': rep['n_folds_sub'],
                'n_folds_space_by_dim': rep['n_folds_space_by_dim'],
                'n_sub': rep['n_sub'],
                'bsz_time': 1000,
                'bsz_space': 1000,
                'seed': rep['seed'],
            }
            if rep.get('path_mask') is not None: 
                flags['path_mask'] = rep['path_mask']
            if args.methods_fse is not None: 
                flags['fse'] = None
            execute_script(path, flags, raise_error)


    if 'allocate-points' in steps:
        print(f"\n{'='*40} POINT ALLOCATION {'='*40}\n")
        for method in args.methods_le:
            print(f"\n{'-'*40} method = {method} {'-'*40}\n")
            for rep in reps:
                print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                path = os.path.join(config.root, 'scripts', 'allocate_points.py')
                flags = {
                    'config': args.config,
                    'dir_out': rep['dir_out'],
                    'dir_out_scratch': rep['dir_out_scratch'],
                    'est_method': method,
                    'world_size': rep['world_size_est'],
                    'seed': rep['seed'],
                }
                execute_script(path, flags, raise_error)


    if 'initialize-loadings' in steps:
        print(f"\n{'='*40} LOADING INITIALIZATION {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")

            # Path and base flags
            path = os.path.join(config.root, 'scripts', f'initialize_loadings.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'n_folds': rep['n_folds_sub'],
                'n_facs': rep['n_facs'][1],
                'prop_init': 1.0,
                'seed': rep['seed'],
            }
            if rep.get('rand_scale') is not None: 
                flags['init_method'] = 'random'
                flags['rand_scale'] = rep['rand_scale']
            else: 
                flags['init_method'] = 'pca_randomized'

            # Full initialization
            flags['split'] = 'full'
            execute_script(path, flags, raise_error)

            # Training initializations
            flags['split'] = 'train'
            for v in range(rep['n_folds_sub']):
                flags['fold'] = v
                execute_script(path, flags, raise_error)


    if 'estimate-loadings' in steps:
        print(f"\n{'='*40} LOADING ESTIMATION {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")

            # Path and base flags
            path = os.path.join(config.root, 'scripts', f'estimate_loads_dssgd.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'world_size': rep['world_size_est'],
                'n_vars': rep['n_vars'],
                'n_facs': rep['n_facs'][1],
                'batch_size': rep.get('dssgd_bsz', 4096),
                'lr': rep.get('dssgd_lr', 64),
                'tol': rep.get('dssgd_tol', 1e-8),
                'patience': 5,
                'max_epochs': rep.get('dssgd_epochs', 1000),
                'seed': rep['seed']
            }

            # Full estimation
            flags['split'] = 'full'
            execute_script(path, flags, raise_error)

            # Training estimations
            flags['split'] = 'train'
            for v in range(rep['n_folds_sub']):
                flags['fold'] = v
                execute_script(path, flags, raise_error)


    if 'rotate' in steps: 
        print(f"\n{'='*40} ROTATE LOADINGS {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', f'rotate.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'est_method': 'dssgd',
            }

            for rot in args.rotations:
                flags['rot_method'] = rot
                
                # Full rotation
                flags['split'] = 'full'
                execute_script(path, flags, raise_error)

                # Training rotation
                flags['file_trg'] = f'model-dssgd-full-{rot}.pth'
                flags['split'] = 'train'
                for v in range(rep['n_folds_sub']):
                    flags['fold'] = v
                    execute_script(path, flags, raise_error)
                del flags['file_trg']
                    

    if 'tune-sigmas' in steps: 
        print(f"\n{'='*40} SIGMA TUNING {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', f'tune_sigmas.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'sigma_grid': [
                    0, 0.5, 0.75, 
                    1.0, 1.25, 1.5, 1.75, 
                    2.0, 2.25, 2.5, 2.75, 
                    3.0, 3.25, 3.5, 3.75, 
                    4.0, 4.25, 4.5, 4.75,
                    5.0, 5.25, 5.5, 5.75,  
                    6.0, 6.25, 6.5, 6.75,  
                ],
                'sz_space': rep['sz_space'],
                'est_method': 'dssgd',
                'n_folds': rep['n_folds_sub'],
            }
            if 'k_wise_sigmas' not in rep:
                flags['universal_param'] = None
            
            for rot in args.rotations:
                flags['rot_method'] = rot
                execute_script(path, flags, raise_error)


    if 'smooth-loadings' in steps: 
        print(f"\n{'='*40} SMOOTH LOADINGS {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', f'smooth_loads.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'sz_space': rep['sz_space'],
                'est_method': 'dssgd',
            }

            # Full smoothing
            flags['split'] = 'full'
            for rot in args.rotations:
                flags['rot_method'] = rot
                flags['path_sigmas'] = os.path.join(rep['dir_out'], f'sigmas-dssgd-{rot}.pt')
                execute_script(path, flags, raise_error)

            # Training smoothing
            flags['split'] = 'train'
            for v in range(rep['n_folds_sub']):
                flags['fold'] = v
                for rot in args.rotations:
                    flags['rot_method'] = rot
                    execute_script(path, flags, raise_error)

    
    if 'tune-kappas' in steps: 
        print(f"\n{'='*40} KAPPA TUNING {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', f'tune_kappas.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'est_method': 'dssgd',
                'n_folds': rep['n_folds_sub'],
                'method': 'sequential' if 'k_wise_kappas' in rep.keys() else 'universal',
                'radius': 1,
            }
            for rot in args.rotations:
                flags['rot_method'] = rot
                execute_script(path, flags, raise_error)


    if 'shrink-loadings' in steps: 
        print(f"\n{'='*40} SHRINK LOADINGS {'='*40}\n")
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'scripts', f'shrink_loads.py')
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'est_method': 'dssgd',
            }

            for rot in args.rotations:
                flags['rot_method'] = rot
                flags['path_kappas'] = os.path.join(rep['dir_out'], f'kappas-dssgd-{rot}.pt')

                # Full smoothing
                flags['split'] = 'full'
                execute_script(path, flags, raise_error)

                # Training smoothing
                flags['split'] = 'train'
                for v in range(rep['n_folds_sub']):
                    flags['fold'] = v
                    execute_script(path, flags, raise_error)


    # ------------------------------------------------------- #
    # -------------------- START MELODIC -------------------- #
    # ------------------------------------------------------- #


    if 'melodic-data-prep' in steps: 
        print(f"\n{'='*40} MELODIC DATA PREP {'='*40}\n")
        path = os.path.join(config.root, 'scripts', 'melodic_data_prep.py')
        errors = {}
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")

            flags = {
                'config': args.config,
                'dir_out_scratch': rep['dir_out_scratch'],
                'sz_space': rep['sz_space'],
                'n_sub': rep['n_sub'],
            }
            code = execute_script(path, flags, raise_error)
            if code != 0:
                if code in errors: 
                    errors[code].append(rep['id'])
                else: 
                    errors[code] = [rep['id']]
        print(f"errors = {errors}")


    if 'melodic-tune-sigma' in steps: 
        print(f"\n{'='*40} MELODIC TUNE SIGMA {'='*40}\n")
        path = os.path.join(config.root, 'scripts', 'melodic_tune_sigma.py')
        errors = {}
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")

            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'n_folds': rep['n_folds_sub'],
                'n_comps': rep['n_facs'][1],
                'fixed_norm': torch.norm(torch.load(os.path.join(rep['dir_out'], 'init-loads-full.pt'))).item(),
            }
            code = execute_script(path, flags, False)  # we anticipate some sigma tuning errors, so we don't raise error
            if code != 0:
                if code in errors: 
                    errors[code].append(rep['id'])
                else: 
                    errors[code] = [rep['id']]
        print(f"errors = {errors}")

    
    if 'melodic-estimation' in steps: 
        print(f"\n{'='*40} MELODIC ESTIMATION {'='*40}\n")
        path = os.path.join(config.root, 'scripts', 'melodic_estimation.py')
        errors_nosmooth = {}
        errors_smooth = {}
        for rep in reps:
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")

            # MELODIC without smoothing
            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'n_folds': rep['n_folds_sub'],
                'n_comps': rep['n_facs'][1],
            }
            code = execute_script(path, flags, False)  # we anticipate some melodic errrors, so we don't raise error
            if code != 0:
                if code in errors_nosmooth: 
                    errors_nosmooth[code].append(rep['id'])
                else: 
                    errors_nosmooth[code] = [rep['id']]

            # MELODIC with smoothing
            path_sigma = os.path.join(rep['dir_out'], 'sigma.pt')
            if os.path.exists(path_sigma):
                flags['sigma'] = torch.load(path_sigma).item()
                code = execute_script(path, flags, False)  # we anticipate some melodic errrors, so we don't raise error
                if code != 0:
                    if code in errors_smooth: 
                        errors_smooth[code].append(rep['id'])
                    else: 
                        errors_smooth[code] = [rep['id']]
            else: 
                print(f"No sigma file for {rep['id']}")
        print(f"errors_nosmooth = {errors_nosmooth}")
        print(f"errors_smooth = {errors_smooth}")


    # ------------------------------------------------------- #
    # --------------------- END MELODIC --------------------- #
    # ------------------------------------------------------- #


    if 'tune-gammas' in steps:
        print(f"\n{'='*40} GAMMA TUNING {'='*40}\n")
        for rot in args.rotations:
            for regime in args.regimes_fse: 
                print(f"\n{'-'*40} RBELS | Regime {regime} | {rot} {'-'*40}\n")
                errors = {}
                for rep in reps:
                    print(f"\n{'-'*20} r = {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'scripts', 'tune_gammas.py')
                    flags = {
                        'config': args.config,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'sub_nums': list(range(3)),
                        'n_folds': multiply_list(rep['n_folds_space_by_dim']),
                        'est_method_loads': 'dssgd',  # only one LE method passed
                        'rot_method': rot,
                        'regime': regime,
                        'batch_size': 500,  # batch data and inv err cov in space
                        'share_across_subs': None,
                        'share_across_k': None,
                        'n_sub_in_rep': rep['n_sub'],
                    }
                    if rep.get('path_mask'):
                        flags['path_mask'] = rep['path_mask']
                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")


    if 'estimate-factor-scores' in steps:
        print(f"\n{'='*40} FACTOR SCORE ESTIMATION {'='*40}\n")
        for method in args.methods_fse: 
            for rot in args.rotations:
                for regime in args.regimes_fse: 
                    print(f"\n{'-'*40} {method.upper()} | Regime {regime} {'-'*40}\n")
                    errors = {}
                    for rep in reps:
                        print(f"\n{'-'*20} r = {rep['id']} {'-'*20}\n")
                        path = os.path.join(config.root, 'scripts', 'estimate_factor_scores.py')
                        flags = {
                            'config': args.config,
                            'dir_out': rep['dir_out'],
                            'dir_out_scratch': rep['dir_out_scratch'],
                            'est_method': method,
                            'sub_nums': list(range(rep['n_sub'])),
                            'split': 'full',
                            'est_method_loads': 'dssgd',
                            'rot_method': rot,
                            'regime': regime,
                            'batch_size': 500,  # batch data and inv err cov in space
                            'agg_subs': None,
                        }
                        if method == 'rbels':
                            path_gamma = os.path.join(
                                rep['dir_out'],
                                f'gammas_rot-{rot}_reg-{regime}.pt'
                            )
                            flags['path_gamma'] = path_gamma
                        if rep.get('path_mask'):
                            flags['path_mask'] = rep['path_mask']

                        code = execute_script(path, flags, raise_error)
                        if code != 0:
                            if code in errors: 
                                errors[code].append(rep['id'])
                            else: 
                                errors[code] = [rep['id']]
                    print(f"errors = {errors}")
        

