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
    return paths


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--fse', action='store_true')
    parser.add_argument(
        '--le_methods', type=str, nargs='+',
        choices=['lbfgs', 'dsgd', 'dssgd'],
        default=['lbfgs', 'dsgd', 'dssgd']
    )
    parser.add_argument(
        '--fse_methods', type=str, nargs='+', 
        choices=['pls', 'pgls', 'rbels', 'rbegls']
    )
    parser.add_argument(
        '--fse_regimes', type=int, nargs='+',
        choices=[1, 2, 3]
    )
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized'],
        default='pca_randomized'
    )
    parser.add_argument('--no_load_smoothing', action='store_true')
    parser.add_argument('--no_fac_smoothing', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--silent_fail', action='store_true')
    args = parser.parse_args()

    # Validate command-line arguments
    if args.fse:
        assert args.fse_methods is not None, "Must specify FSE methods!"
        assert args.fse_regimes is not None, "Must pass FSE regimes!"
        assert len(args.le_methods) == 1, "Must pass only one LE method!"

    config = load_config(args.config)
    raise_error = not args.silent_fail

    # Setup simulation files
    path = os.path.join(config.root, 'setup_simulations.py')
    flags = {
        'config': args.config,
        'design': args.design,
    }
    if args.benchmark: 
        flags['benchmark'] = None
    if args.fse:
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


    print(f"\n{'='*40} DATA SIMULATION {'='*40}\n")
    for rep in reps: 
        print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
        path = os.path.join(config.root, 'simulate_data.py')
        flags = {
            'config': args.config,
            'dir_dataset': rep['dir_dataset'],
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
        if rep['id'].endswith('-0'):  # Write true loads, facs, errs
            flags['dir_truth'] = rep['dir_out_sim']
        execute_script(path, flags, raise_error)


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
            'bsz_time': 50,
            'bsz_space': 100,
            'seed': rep['seed'],
        }
        if args.fse: 
            flags['fse'] = None
        execute_script(path, flags, raise_error)


    print(f"\n{'='*40} POINT ALLOCATION {'='*40}\n")
    for method in args.le_methods:
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
                'init_method': args.init_method,
                'prop_init': 1.0,
                'seed': rep['seed'],
            }
            execute_script(path, flags, raise_error)


    if not args.no_load_smoothing:
        print(f"\n{'='*40} ALPHA TUNING {'='*40}\n")
        for method in args.le_methods:
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
                            'lr': 0.1,
                            'history_size': 10,
                            'tol': 1e-6,
                            'patience': 5,
                            'max_epochs': 5000,
                        }

                    if method == 'dsgd':
                        flags = flags | {
                            'world_size': rep['world_size_est'],
                            'batch_size': 128,
                            'lr': 1,
                            'tol': 1e-6,
                            'patience': 5,
                            'max_epochs': 5000,
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


    print(f"\n{'='*40} LOADING ESTIMATION {'='*40}\n")
    for method in args.le_methods:
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
                if args.no_load_smoothing: 
                    flags['alpha'] = 0
                else: 
                    flags['alpha'] = torch.load(
                        os.path.join(rep['dir_out'], f'alpha-{method}.pt')
                    ).item()

            if method == 'lbfgs':
                flags = flags | {
                    'lr': 0.1,
                    'history_size': 10,
                    'tol': 1e-6,
                    'patience': 5,
                    'max_epochs': 5000,
                }

            if method == 'dsgd':
                flags = flags | {
                    'world_size': rep['world_size_est'],
                    'batch_size': 128,
                    'lr': 10,
                    'tol': 1e-6,
                    'patience': 5,
                    'max_epochs': 5000,
                    'seed': rep['seed']
                }

            if method == 'dssgd':
                flags = flags | {
                    'world_size': rep['world_size_est'],
                    'batch_size': 128,
                    'lr': 10,
                    'tol': 1e-6,
                    'patience': 5,
                    'max_epochs': 5000,
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


    if args.fse:

        print(f"\n{'='*40} INVERSE ERROR COVARIANCE ESTIMATION {'='*40}\n")
        if any([m in ['pgls', 'rbegls'] for m in args.fse_methods]):
            for regime in args.fse_regimes: 
                print(f"\n{'-'*40} Regime {regime} {'-'*40}\n")
                errors = {}
                for rep in reps:
                    print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'compute_inv_err_cov.py')
                    flags = {
                        'config': args.config,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'dir_truth': rep['dir_out_sim'],
                        'regime': regime,
                        'split': 'full',
                        'sz_space': rep['sz_space'],
                        'est_method_loads': args.le_methods[0],  # only one LE method passed
                        'delta_true': rep['delta'],
                        'eval_cutoff': 0,  # TODO: Fine tune
                        'phi': 0.0001,  # TODO: Fine tune
                        'batch_size': 100,  # batch inv err cov in space
                    }
                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")

        
        if not args.no_fac_smoothing: 
            print(f"\n{'='*40} GAMMA TUNING {'='*40}\n")
            for method in args.fse_methods: 
                if method in ['rbels', 'rbegls']:
                    for regime in args.fse_regimes: 
                        print(f"\n{'-'*40} {method.upper()} | Regime {regime} {'-'*40}\n")
                        errors = {}
                        for rep in reps:
                            print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                            path = os.path.join(config.root, 'tune_gamma.py')

                            flags = {
                                'config': args.config,
                                'dir_out': rep['dir_out'],
                                'dir_out_scratch': rep['dir_out_scratch'],
                                'dir_truth': rep['dir_out_sim'],
                                'est_method': method,
                                'n_time': rep['n_time'],
                                'est_method_loads': args.le_methods[0],  # only one LE method passed
                                'regime': regime,
                                'batch_size': 100,  # batch data and inv err cov in space
                            }
                            code = execute_script(path, flags, raise_error)
                            if code != 0:
                                if code in errors: 
                                    errors[code].append(rep['id'])
                                else: 
                                    errors[code] = [rep['id']]
                        print(f"errors = {errors}")


        print(f"\n{'='*40} FACTOR SCORE ESTIMATION {'='*40}\n")
        for method in args.fse_methods: 
            for regime in args.fse_regimes: 
                print(f"\n{'-'*40} {method.upper()} | Regime {regime} {'-'*40}\n")
                errors = {}
                for rep in reps:
                    print(f"\n{'-'*20} {rep['id']} {'-'*20}\n")
                    path = os.path.join(config.root, 'estimate_factor_scores.py')

                    flags = {
                        'config': args.config,
                        'dir_out': rep['dir_out'],
                        'dir_out_scratch': rep['dir_out_scratch'],
                        'dir_truth': rep['dir_out_sim'],
                        'est_methods': method,
                        'split': 'full',
                        'est_method_loads': args.le_methods[0],  # only one LE method passed
                        'regime': regime,
                        'batch_size': 100,  # batch data and inv err cov in space
                    }
                    if args.no_fac_smoothing:
                        flags['gamma'] = 0
                    else: 
                        if method in ['rbels', 'rbegls']:
                            flags['gamma'] = torch.load(
                                os.path.join(rep['dir_out'], f'gamma-{method}-r{regime}.pt')
                            ).item()

                    code = execute_script(path, flags, raise_error)
                    if code != 0:
                        if code in errors: 
                            errors[code].append(rep['id'])
                        else: 
                            errors[code] = [rep['id']]
                print(f"errors = {errors}")
        

