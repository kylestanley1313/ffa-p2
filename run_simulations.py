import argparse
import os
import torch
from typing import Dict

from config import load_config
from utils import (
    execute_script, 
    load_yaml, 
)


def get_all_rep_paths(config, design_id): 
    paths = []
    dir_design = os.path.join(config.root, 'designs', design_id)
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
            'tune-alpha',
            'estimate-loadings',
            'compute-inv-err-cov',
            'tune-gamma',
            'estimate-factor-scores',
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
            'compute-inv-err-cov',
            'tune-gamma',
            'estimate-factor-scores',
        ]
    else: 
        steps = args.steps


    # Setup simulation files
    if 'setup-simulations' in steps:
        path = os.path.join(config.root, 'setup_simulations.py')
        flags = {
            'config': args.config,
            'design': args.design,
        }
        if args.benchmark: 
            flags['benchmark'] = None
        if args.methods_fse is not None:
            flags['fse'] = None
        execute_script(path, flags, raise_error)


    # Get all repetitions
    reps = []
    for path in get_all_rep_paths(config, args.design):
        rep = load_yaml(path)
        rep['id'] = (path.split('designs/')[-1]
                     .replace('.yml', '')
                     .replace('/', '_'))
        reps.append(rep)


    if 'simulate-data' in steps:
        print(f"\n{'='*40} DATA SIMULATION {'='*40}\n")
        for rep in reps: 
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'simulate_data.py')
            flags = {
                'config': args.config,
                'dir_dataset': rep['dir_dataset'],
                'dir_dataset_truth': rep['dir_dataset_truth'],
                'dir_out': rep['dir_out'],
                'n_time': rep['n_time'],
                'sz_space': rep['sz_space'],
                'factor_kernel_length': rep['factor_kernel_length'],
                'load_scheme': rep['load_scheme'],
                'err_scheme': rep['err_scheme'],
                'n_facs': rep['n_facs'],
                'delta': rep['delta'],
                'prop_global': rep['prop_global'],
                'batch_size': 100,
                'seed': rep['seed'],
            }
            if not rep['id'].endswith('-0'):  # Read true loads and errs
                flags['read_loads'] = None
                flags['read_errs'] = None
            execute_script(path, flags, raise_error)


    if 'compute-covariance' in steps:
        print(f"\n{'='*40} COVARIANCE COMPUTATION {'='*40}\n")
        for rep in reps: 
            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
            path = os.path.join(config.root, 'compute_covariance.py')
            flags = {
                'config': args.config,
                'dir_dataset': rep['dir_dataset'],
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'world_size': rep['world_size_cov'],
                'delta': rep['delta_est'],
                'prop_train_time': 0.8,
                'prop_train_space': 0.8,
                'bsz_time': 500,
                'bsz_space': 1000,
                'seed': rep['seed'],
            }
            if args.methods_fse is not None: 
                flags['fse'] = None
            execute_script(path, flags, raise_error)


    if 'allocate-points' in steps:
        print(f"\n{'='*40} POINT ALLOCATION {'='*40}\n")
        for method in args.methods_le:
            print(f"\n{'-'*40} method = {method} {'-'*40}\n")
            for rep in reps:
                print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                path = os.path.join(config.root, 'allocate_points.py')
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
        for split in ['full', 'train', 'valid']: 
            print(f"\n{'-'*40} split = {split} {'-'*40}\n")
            for rep in reps:
                print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                path = os.path.join(config.root, f'initialize_loadings.py')
                flags = {
                    'config': args.config,
                    'dir_out': rep['dir_out'],
                    'dir_out_scratch': rep['dir_out_scratch'],
                    'split': split,
                    'sz_space': rep['sz_space'],
                    'n_facs': rep['n_facs'],
                    'init_method': 'pca_randomized',
                    'prop_init': 1.0,
                    'seed': rep['seed'],
                }
                execute_script(path, flags, raise_error)


    if 'tune-alpha' in steps:
        print(f"\n{'='*40} ALPHA TUNING {'='*40}\n")
        for method in args.methods_le:
            if method in ['lbfgs', 'dsgd']:
                print(f"\n{'-'*40} {method.upper()} {'-'*40}\n")
                errors = {}
                for rep in reps: 
                    print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'tune_alpha.py')
                    
                    flags = {
                        'config': args.config,
                        'est_method': method,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'sz_space': rep['sz_space'],
                        'n_facs': rep['n_facs'],
                    }

                    if method == 'lbfgs':
                        flags = flags | {
                            'lr': 1.0,
                            'history_size': 10,
                            'tol': 1e-6,
                            'patience': 5,
                            'max_epochs': 1000,
                        }

                    if method == 'dsgd':
                        flags = flags | {
                            'world_size': rep['world_size_est'],
                            'batch_size': 1024,
                            'lr': 4.0,
                            'tol': 1e-6,
                            'patience': 5,
                            'max_epochs': 1000,
                            'seed': rep['seed']
                        }

                    if args.benchmark and method == 'dsgd':
                        flags['benchmark'] = None
                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")


    if 'estimate-loadings' in steps:
        print(f"\n{'='*40} LOADING ESTIMATION {'='*40}\n")
        for method in args.methods_le:
            print(f"\n{'-'*40} {method.upper()} {'-'*40}\n")
            errors = {}
            for rep in reps: 
                print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                path = os.path.join(config.root, f'estimate_loads_{method}.py')

                flags = {
                    'config': args.config,
                    'dir_out': rep['dir_out'],
                    'dir_out_scratch': rep['dir_out_scratch'],
                    'split': 'full',
                    'sz_space': rep['sz_space'],
                    'n_facs': rep['n_facs'],
                }

                if method in ['lbfgs', 'dsgd']:
                    try: 
                        flags['alpha'] = torch.load(
                            os.path.join(rep['dir_out'], f'alpha-{method}.pt')
                        ).item()
                    except FileNotFoundError:
                        flags['alpha'] = 0

                if method == 'lbfgs':
                    flags = flags | {
                        'lr': 1.0,
                        'history_size': 10,
                        'tol': 1e-6,
                        'patience': 5,
                        'max_epochs': 1000,
                    }

                if method == 'dsgd':
                    flags = flags | {
                        'world_size': rep['world_size_est'],
                        'batch_size': 1024,
                        'lr': 4.0,
                        'tol': 1e-6,
                        'patience': 5,
                        'max_epochs': 1000,
                        'seed': rep['seed']
                    }

                if method == 'dssgd':
                    flags = flags | {
                        'world_size': rep['world_size_est'],
                        'batch_size': 4096,
                        'lr': 16.0,
                        'tol': 1e-6,
                        'patience': 5,
                        'max_epochs': 1000,
                        'seed': rep['seed']
                    }

                if args.benchmark:
                    flags['benchmark'] = None
                code = execute_script(path, flags, raise_error)
                if code != 0:
                    if code in errors: 
                        errors[code].append(rep['id'])
                    else: 
                        errors[code] = [rep['id']]
            print(f"errors = {errors}")


    if 'compute-inv-err-cov' in steps:
        print(f"\n{'='*40} INVERSE ERROR COVARIANCE ESTIMATION {'='*40}\n")
        if any([m in ['pgls', 'rbegls'] for m in args.methods_fse]):
            for regime in args.regimes_fse: 
                print(f"\n{'-'*40} Regime {regime} {'-'*40}\n")
                errors = {}
                for rep in reps:
                    print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'compute_inv_err_cov.py')
                    flags = {
                        'config': args.config,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'regime': regime,
                        'split': 'full',
                        'sz_space': rep['sz_space'],
                        'est_method_loads': args.methods_le[0],  # only one LE method passed
                        'delta': rep['delta_est'],
                        'eval_cutoff': 0 if regime == 1 else 0.5,  # TODO: Fine tune
                        'phi': 0.001,  # TODO: Fine tune (0.000001 unstable when M = 40)
                        'batch_size': 100,  # batch inv err cov in space
                    }
                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")

        
    if 'tune-gamma' in steps:
        print(f"\n{'='*40} GAMMA TUNING {'='*40}\n")
        for method in args.methods_fse: 
            if method in ['rbels', 'rbegls']:
                for regime in args.regimes_fse: 
                    print(f"\n{'-'*40} {method.upper()} | Regime {regime} {'-'*40}\n")
                    errors = {}
                    for rep in reps:
                        print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                        path = os.path.join(config.root, 'tune_gamma.py')

                        flags = {
                            'config': args.config,
                            'dir_out': rep['dir_out'],
                            'dir_out_scratch': rep['dir_out_scratch'],
                            'est_method': method,
                            'n_time': rep['n_time'],
                            'est_method_loads': args.methods_le[0],  # only one LE method passed
                            'regime': regime,
                            'batch_size': 100,  # batch data and inv err cov in space
                            'skip_batching': None,  # This is okay for relatively small n_space
                        }
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
            for regime in args.regimes_fse: 
                print(f"\n{'-'*40} {method.upper()} | Regime {regime} {'-'*40}\n")
                errors = {}
                for rep in reps:
                    print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'estimate_factor_scores.py')

                    flags = {
                        'config': args.config,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'est_methods': method,
                        'split': 'full',
                        'est_method_loads': args.methods_le[0],  # only one LE method passed
                        'regime': regime,
                        'batch_size': 100,  # batch data and inv err cov in space
                    }
                    if method in ['rbels', 'rbegls']:
                        path_gamma = os.path.join(rep['dir_out'], f'gamma-{method}-r{regime}.pt')
                        if os.path.exists(path_gamma):
                            flags['gamma'] = torch.load(path_gamma).item()
                        else: 
                            print("No gamma file found!")
                            flags['gamma'] = 0

                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")
        

