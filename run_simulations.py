import argparse
import os
import torch
from typing import Dict

from config import load_config
from utils.utils import execute_script, load_yaml, refresh_directory


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
    parser.add_argument(
        '--est_methods', type=str, nargs='+', 
        default=['lbfgs', 'dsgd', 'dssgd']
    )
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--silent_fail', action='store_true')
    args = parser.parse_args()

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
    execute_script(path, flags, raise_error)

    # Get all repetitions
    reps = []
    for path in get_all_rep_paths(config, args.design):
        rep = load_yaml(path)
        rep['id'] = (path.split('designs/')[-1]
                     .replace('.yml', '')
                     .replace('/', '_'))
        reps.append(rep)


    print(f"\n{'====='*8} DATA SIMULATION {'====='*8}\n")
    for rep in reps: 
        print(f"\n{'-----'*4} {rep['id']} {'-----'*4}\n")
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
            'batch_size': 1000,
            'seed': rep['seed'],
        }
        if rep['id'].endswith('-0'):  # Write true loads, facs, errs
            flags['dir_truth'] = rep['dir_out_sim']
        execute_script(path, flags, raise_error)


    print(f"\n{'====='*8} DATA PREPARATION {'====='*8}\n")
    for method in args.est_methods:
        print(f"\n{'-----'*8} method = {method} {'-----'*8}\n")
        for rep in reps:
            print(f"\n{'-----'*4} {rep['id']} {'-----'*4}\n")
            path = os.path.join(config.root, 'prepare_data.py')
            flags = {
                'config': args.config,
                'dir_dataset': rep['dir_dataset'],
                'est_method': method,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'world_size_est': rep['world_size_est'],
                'world_size_cov': rep['world_size_cov'],
                'delta': rep['delta_est'],
                'prop_train': rep['prop_train'],
                'seed': rep['seed'],
            }
            if args.benchmark: 
                flags['benchmark'] = None
            execute_script(path, flags, raise_error)


    print(f"\n{'====='*8} LOADING INITIALIZATION {'====='*8}\n")
    for split in ['full', 'train', 'valid']: 
        print(f"\n{'-----'*8} split = {split} {'-----'*8}\n")
        for rep in reps:
            print(f"\n{'-----'*4} {rep['id']} {'-----'*4}\n")
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
            if args.benchmark:
                flags['benchmark'] = None
            execute_script(path, flags, raise_error)
                

    print(f"\n{'====='*8} ALPHA TUNING {'====='*8}\n")
    for est_method in args.est_methods:
        if est_method in ['lbfgs', 'dsgd']:
            print(f"\n{'-----'*8} {est_method.upper()} {'-----'*8}\n")
            for rep in reps: 
                print(f"\n{'-----'*4} {rep['id']} {'-----'*4}\n")
                path = os.path.join(config.root, 'tune_alpha.py')
                
                flags = {
                    'config': args.config,
                    'est_method': est_method,
                    'dir_out': rep['dir_out'],
                    'dir_out_scratch': rep['dir_out_scratch'],
                    'sz_space': rep['sz_space'],
                    'n_facs': rep['n_facs'],
                }

                if est_method == 'lbfgs':
                    flags = flags | {
                        'lr': 0.1,
                        'history_size': 10,
                        'tol': 1e-7,
                        'patience': 5,
                        'max_epochs': 5000,
                    }

                if est_method == 'dsgd':
                    flags = flags | {
                        'world_size': rep['world_size_est'],
                        'batch_size': 128,
                        'lr': 1,
                        'tol': 1e-7,
                        'patience': 5,
                        'max_epochs': 5000,
                        'seed': rep['seed']
                    }

                if args.benchmark and est_method == 'dsgd':
                    flags['benchmark'] = None
                execute_script(path, flags, raise_error)


    print(f"\n{'====='*8} ESTIMATION {'====='*8}\n")
    for est_method in args.est_methods:
        print(f"\n{'-----'*8} {est_method.upper()} {'-----'*8}\n")
        for rep in reps: 
            print(f"\n{'-----'*4} {rep['id']} {'-----'*4}\n")
            path = os.path.join(config.root, f'estimate_loads_{est_method}.py')

            flags = {
                'config': args.config,
                'dir_out': rep['dir_out'],
                'dir_out_scratch': rep['dir_out_scratch'],
                'split': 'full',
                'sz_space': rep['sz_space'],
                'n_facs': rep['n_facs'],
            }

            if est_method in ['lbfgs', 'dsgd']:
                flags['alpha'] = torch.load(
                    os.path.join(rep['dir_out'], f'alpha-{est_method}.pt')
                ).item()

            if est_method == 'lbfgs':
                flags = flags | {
                    'lr': 0.1,
                    'history_size': 10,
                    'tol': 1e-7,
                    'patience': 5,
                    'max_epochs': 5000,
                }

            if est_method == 'dsgd':
                flags = flags | {
                    'world_size': rep['world_size_est'],
                    'batch_size': 128,
                    'lr': 1,
                    'tol': 1e-7,
                    'patience': 5,
                    'max_epochs': 5000,
                    'seed': rep['seed']
                }

            if est_method == 'dssgd':
                flags = flags | {
                    'world_size': rep['world_size_est'],
                    'batch_size': 128,
                    'lr': 1,
                    'tol': 1e-7,
                    'patience': 5,
                    'max_epochs': 5000,
                    'seed': rep['seed']
                }

            if args.benchmark and est_method in ['dsgd', 'dssgd']:
                flags['benchmark'] = None
            execute_script(path, flags, raise_error)





    # # Define design directories
    # dir_dataset_design = os.path.join(config.scratch_root, 'datasets', args.design)
    # dir_out_design = os.path.join(config.root, 'out', args.design)
    # dir_out_design_scratch = os.path.join(config.scratch_root, 'out', args.design)
    # dir_design = os.path.join(config.root, 'designs', args.design)

    # # Run repetitions for each simulation
    # sim_ids = os.listdir(dir_design)
    # for sim_id in sim_ids:
    #     print(f"\n{'====='*8} {sim_id} {'====='*8}\n")

    #     # Define simulation directories
    #     dir_dataset_sim = os.path.join(dir_dataset_design, sim_id)
    #     dir_out_sim = os.path.join(dir_out_design, sim_id)
    #     dir_out_sim_scratch = os.path.join(dir_out_design_scratch, sim_id)
    #     dir_sim = os.path.join(dir_design, sim_id)

    #     rep_ids = [f.split('.')[0] for f in os.listdir(dir_sim)]
    #     for rep_id in rep_ids[:1]:
    #         print(f"\n{'-----'*8} {rep_id} {'-----'*8}\n")

    #         # Define repetition directories
    #         dir_dataset_rep = os.path.join(dir_dataset_sim, rep_id)
    #         dir_out_rep = os.path.join(dir_out_sim, rep_id)
    #         dir_out_rep_scratch = os.path.join(dir_out_sim_scratch, rep_id)

    #         # Load repetition YAML
    #         path = os.path.join(dir_sim, f'{rep_id}.yml')
    #         repetition = load_yaml(path)

    #         print(f"\n{'-----'*4} DATA SIMULATION {'-----'*4}\n")
    #         path = os.path.join(config.root, 'simulate_data.py')
    #         flags = {
    #             'config': args.config,
    #             'dir_dataset': dir_dataset_rep,
    #             'dir_out': dir_out_rep,
    #             'n_time': repetition['n_time'],
    #             'sz_space': repetition['sz_space'],
    #             'factor_kernel_length': repetition['factor_kernel_length'],
    #             'load_scheme': repetition['load_scheme'],
    #             'err_scheme': repetition['err_scheme'],
    #             'n_facs': repetition['n_facs'],
    #             'delta': repetition['delta'],
    #             'prop_global': repetition['prop_global'],
    #             'batch_size': 1000,
    #             'seed': repetition['seed'],
    #         }
    #         if rep_id == 'rep-0':  # Write true loads, facs, errs
    #             flags['dir_truth'] = dir_out_sim
    #         execute_script(path, flags, raise_error)

    #         print(f"\n{'-----'*4} DATA PREPARATION {'-----'*4}\n")
    #         for est_method in args.est_methods:
    #             print(f"PREPARING FOR {est_method.upper()}...")
    #             path = os.path.join(config.root, 'prepare_data.py')
    #             flags = {
    #                 'config': args.config,
    #                 'dir_dataset': dir_dataset_rep,
    #                 'est_method': est_method,
    #                 'dir_out': dir_out_rep,
    #                 'dir_out_scratch': dir_out_rep_scratch,
    #                 'world_size_est': repetition['world_size_est'],
    #                 'world_size_cov': repetition['world_size_cov'],
    #                 'delta': repetition['delta'],
    #                 'prop_train': repetition['prop_train'],
    #                 'seed': repetition['seed'],
    #             }
    #             if args.benchmark: 
    #                 flags['benchmark'] = None
    #             execute_script(path, flags, raise_error)

    #         # print(f"\n{'-----'*4} ALPHA TUNING {'-----'*4}\n")
    #         # path = os.path.join(config.root, 'tune_alpha.py')
    #         # flags = {
    #         #     'config': args.config,
    #         #     'dir_out': dir_out_rep,
    #         #     'dir_out_scratch': dir_out_rep_scratch,
    #         #     'world_size': repetition['world_size_est'],
    #         #     'grid_shape': repetition['grid_shape'],
    #         #     'num_facs': repetition['num_facs'],
    #         #     'delta': repetition['delta'],
    #         #     'init_method': repetition['init_method'],
    #         #     'init_prop': repetition['init_prop'],
    #         #     'batch_size': repetition['batch_size'],
    #         #     'lr': repetition['lr'],
    #         #     'max_epochs': repetition['max_epochs'],
    #         #     'seed': repetition['seed'],
    #         # }
    #         # execute_script(path, flags, raise_error)
    #         # path = os.path.join(dir_out_rep, 'alpha.pt')
    #         # alpha = torch.load(path).item()

    #         print(f"\n{'-----'*4} ESTIMATION {'-----'*4}\n")
    #         for est_method in args.est_methods:
    #             print(f"ESTIMATING VIA {est_method.upper()}")
    #             path = os.path.join(config.root, f'estimate_loads_{est_method}.py')
    #             flags = {
    #                 'config': args.config,
    #                 'dir_out': dir_out_rep,
    #                 'dir_out_scratch': dir_out_rep_scratch,
    #                 'world_size': repetition['world_size_est'],
    #                 'split': 'full',
    #                 'sz_space': repetition['sz_space'],
    #                 'n_facs': repetition['n_facs'],
    #                 # 'alpha': alpha,
    #                 'delta': repetition['delta_est'],
    #                 'init_method': repetition['init_method'],
    #                 'prop_init': repetition['prop_init'],
    #                 'batch_size': repetition['batch_size'],
    #                 'lr': repetition['lr'],
    #                 'max_epochs': repetition['max_epochs'],
    #                 'seed': repetition['seed'],
    #             }
    #             if args.benchmark:
    #                 flags['benchmark'] = None
    #             execute_script(path, flags, raise_error)

    #         # print(f"\n{'-----'*4} ROTATION {'-----'*4}\n")
    #         # path = os.path.join(config.root, 'rotate.py')
    #         # flags = {
    #         #     'config': args.config,
    #         #     'dir_out': dir_out_rep
    #         # }
    #         # execute_script(path, flags, raise_error)

    #         # print(f"\n{'-----'*4} SHRINKAGE {'-----'*4}\n")
    #         # path = os.path.join(config.root, 'shrink.py')
    #         # flags = {
    #         #     'config': args.config,
    #         #     'dir_out': dir_out_rep
    #         # }
    #         # execute_script(path, flags, raise_error)

    #         # print(f"\n{'-----'*4} EVALUATION {'-----'*4}\n")


    # exit(0)

    # # Run each simulation
    # dir_design = os.path.join(config.root, 'designs', args.design)
    # sim_ids = [f.split('.')[0] for f in os.listdir(dir_design)]
    # for sim_id in sim_ids:

    #     print(f"\n========== {sim_id} ==========\n")

    #     # Load simulation
    #     path = os.path.join(dir_design, f'{sim_id}.yml')
    #     simulation = load_yaml(path)
    #     dir_out = os.path.join(args.design, sim_id)

    #     # ----- DATA SIMULATION ----- #
    #     path = os.path.join(config.root, 'simulate_data.py')
    #     flags = {
    #         'config': args.config,
    #         'dir': sim_id,
    #         'grid_shape': simulation['grid_shape'],
    #         'load_scheme': simulation['load_scheme'],
    #         'err_scheme': simulation['err_scheme'],
    #         'num_samps': simulation['num_samps'],
    #         'batch_size': int(simulation['num_samps'] / 4)
    #     }
    #     execute_script(path, flags, raise_error)

    #     # ----- DATA PREPARATION ----- #
    #     path = os.path.join(config.root, 'prepare_data.py')
    #     flags = {
    #         'config': args.config,
    #         'dataset': sim_id,
    #         'est_method': simulation['estimation'],
    #         'dir_out': dir_out,
    #         'world_size_est': simulation['world_size_est'],
    #         'world_size_cov': simulation['world_size_cov'],
    #         'delta': simulation['delta'],
    #         'train_prop': simulation['train_prop']
    #     }
    #     execute_script(path, flags, raise_error)

    #     # ------ ESTIMATION ----- #
    #     path = os.path.join(config.root, f'factor_model_{simulation["estimation"]}.py')
    #     flags = {
    #         'config': args.config,
    #         'dir_out': dir_out,
    #         'world_size': simulation['world_size_est'],
    #         'grid_shape': simulation['grid_shape'],
    #         'num_facs': simulation['num_facs'],
    #         'delta': simulation['delta'],
    #         'init_method': simulation['init_method'],
    #         'init_prop': simulation['init_prop'],
    #         'batch_size': simulation['batch_size'],
    #         'lr': simulation['lr'],
    #         'max_epochs': simulation['max_epochs']
    #     }
    #     if simulation['estimation'] == 'ddp':
    #         flags['alpha'] = simulation['alpha']
    #     # if simulation['opt']: 
    #     #     flags['opt'] = None
    #     execute_script(path, flags, raise_error)


    #     # ------ POST-PROCESSING ------ #
    #     path = os.path.join(config.root, 'postprocess.py')
    #     flags = {
    #         'config': args.config,
    #         'dir_out': dir_out
    #     }
    #     execute_script(path, flags, raise_error)


    #     # ------ EVALUATION ----- #
    #     path = os.path.join(config.root, 'evaluate_model.py')
    #     flags = {
    #         'dir_out': dir_out,
    #         'grid_shape': simulation['grid_shape'],
    #         'plot': None
    #     }
    #     if config.benchmark:
    #         flags['benchmark'] = None
    #     execute_script(path, flags, raise_error)
