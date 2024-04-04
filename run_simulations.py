import argparse
import os
import torch

from config import load_config
from utils.utils import execute_script, load_yaml, refresh_directory


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--silent_fail', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    raise_error = not args.silent_fail

    # Setup simulation files
    path = os.path.join(config.root, 'setup_simulations.py')
    flags = {
        'config': args.config,
        'design': args.design
    }
    execute_script(path, flags, raise_error)

    # Define design directories
    dir_dataset_design = os.path.join(config.scratch_root, 'datasets', args.design)
    dir_out_design = os.path.join(config.root, 'out', args.design)
    dir_out_design_scratch = os.path.join(config.scratch_root, 'out', args.design)
    dir_design = os.path.join(config.root, 'designs', args.design)

    # Run repetitions for each simulation
    sim_ids = os.listdir(dir_design)
    for sim_id in sim_ids:
        print(f"\n{'====='*8} {sim_id} {'====='*8}\n")

        # Define simulation directories
        dir_dataset_sim = os.path.join(dir_dataset_design, sim_id)
        dir_out_sim = os.path.join(dir_out_design, sim_id)
        dir_out_sim_scratch = os.path.join(dir_out_design_scratch, sim_id)
        dir_sim = os.path.join(dir_design, sim_id)

        rep_ids = [f.split('.')[0] for f in os.listdir(dir_sim)]
        for rep_id in rep_ids:
            print(f"\n{'-----'*8} {rep_id} {'-----'*8}\n")

            # Define repetition directories
            dir_dataset_rep = os.path.join(dir_dataset_sim, rep_id)
            dir_out_rep = os.path.join(dir_out_sim, rep_id)
            dir_out_rep_scratch = os.path.join(dir_out_sim_scratch, rep_id)

            # Load repetition YAML
            path = os.path.join(dir_sim, f'{rep_id}.yml')
            repetition = load_yaml(path)

            print(f"\n{'-----'*4} DATA SIMULATION {'-----'*4}\n")
            path = os.path.join(config.root, 'simulate_data.py')
            flags = {
                'config': args.config,
                'dir': dir_dataset_rep,
                'grid_shape': repetition['grid_shape'],
                'load_scheme': repetition['load_scheme'],
                'err_scheme': repetition['err_scheme'],
                'num_samps': repetition['num_samps'],
                'batch_size': int(repetition['num_samps'] / 4)
            }
            execute_script(path, flags, raise_error)

            print(f"\n{'-----'*4} DATA PREPARATION {'-----'*4}\n")
            path = os.path.join(config.root, 'prepare_data.py')
            flags = {
                'config': args.config,
                'dir_dataset': dir_dataset_rep,
                'est_method': repetition['estimation'],
                'dir_out': dir_out_rep,
                'dir_out_scratch': dir_out_rep_scratch,
                'world_size_est': repetition['world_size_est'],
                'world_size_cov': repetition['world_size_cov'],
                'delta': repetition['delta'],
                'train_prop': repetition['train_prop']
            }
            execute_script(path, flags, raise_error)

            print(f"\n{'-----'*4} ALPHA TUNING {'-----'*4}\n")
            path = os.path.join(config.root, 'tune_alpha.py')
            flags = {
                'config': args.config,
                'dir_out': dir_out_rep,
                'dir_out_scratch': dir_out_rep_scratch,
                'world_size': repetition['world_size_est'],
                'grid_shape': repetition['grid_shape'],
                'num_facs': repetition['num_facs'],
                'delta': repetition['delta'],
                'init_method': repetition['init_method'],
                'init_prop': repetition['init_prop'],
                'batch_size': repetition['batch_size'],
                'lr': repetition['lr'],
                'max_epochs': repetition['max_epochs'],
                'seed': repetition['seed'],
            }
            execute_script(path, flags, raise_error)
            path = os.path.join(dir_out_rep, 'alpha.pt')
            alpha = torch.load(path).item()

            print(f"\n{'-----'*4} ESTIMATION (alpha = {alpha}) {'-----'*4}\n")
            path = os.path.join(config.root, 'factor_model_ddp.py')  # TODO: STRAT handling
            flags = {
                'config': args.config,
                'dir_out': dir_out_rep,
                'dir_out_scratch': dir_out_rep_scratch,
                'world_size': repetition['world_size_est'],
                'split': 'full',
                'grid_shape': repetition['grid_shape'],
                'num_facs': repetition['num_facs'],
                'alpha': alpha,
                'delta': repetition['delta'],
                'init_method': repetition['init_method'],
                'init_prop': repetition['init_prop'],
                'batch_size': repetition['batch_size'],
                'lr': repetition['lr'],
                'max_epochs': repetition['max_epochs'],
                'seed': repetition['seed'],
            }
            execute_script(path, flags, raise_error)

            print(f"\n{'-----'*4} ROTATION {'-----'*4}\n")
            path = os.path.join(config.root, 'rotate.py')  # TODO: STRAT handling
            flags = {
                'config': args.config,
                'dir_out': dir_out_rep
            }
            execute_script(path, flags, raise_error)

            print(f"\n{'-----'*4} KAPPA TUNING {'-----'*4}\n")

            print(f"\n{'-----'*4} SHRINKAGE {'-----'*4}\n")

            print(f"\n{'-----'*4} EVALUATION {'-----'*4}\n")


    exit(0)

    # Run each simulation
    dir_design = os.path.join(config.root, 'designs', args.design)
    sim_ids = [f.split('.')[0] for f in os.listdir(dir_design)]
    for sim_id in sim_ids:

        print(f"\n========== {sim_id} ==========\n")

        # Load simulation
        path = os.path.join(dir_design, f'{sim_id}.yml')
        simulation = load_yaml(path)
        dir_out = os.path.join(args.design, sim_id)

        # ----- DATA SIMULATION ----- #
        path = os.path.join(config.root, 'simulate_data.py')
        flags = {
            'config': args.config,
            'dir': sim_id,
            'grid_shape': simulation['grid_shape'],
            'load_scheme': simulation['load_scheme'],
            'err_scheme': simulation['err_scheme'],
            'num_samps': simulation['num_samps'],
            'batch_size': int(simulation['num_samps'] / 4)
        }
        execute_script(path, flags, raise_error)

        # ----- DATA PREPARATION ----- #
        path = os.path.join(config.root, 'prepare_data.py')
        flags = {
            'config': args.config,
            'dataset': sim_id,
            'est_method': simulation['estimation'],
            'dir_out': dir_out,
            'world_size_est': simulation['world_size_est'],
            'world_size_cov': simulation['world_size_cov'],
            'delta': simulation['delta'],
            'train_prop': simulation['train_prop']
        }
        execute_script(path, flags, raise_error)

        # ------ ESTIMATION ----- #
        path = os.path.join(config.root, f'factor_model_{simulation["estimation"]}.py')
        flags = {
            'config': args.config,
            'dir_out': dir_out,
            'world_size': simulation['world_size_est'],
            'grid_shape': simulation['grid_shape'],
            'num_facs': simulation['num_facs'],
            'delta': simulation['delta'],
            'init_method': simulation['init_method'],
            'init_prop': simulation['init_prop'],
            'batch_size': simulation['batch_size'],
            'lr': simulation['lr'],
            'max_epochs': simulation['max_epochs']
        }
        if simulation['estimation'] == 'ddp':
            flags['alpha'] = simulation['alpha']
        # if simulation['opt']: 
        #     flags['opt'] = None
        execute_script(path, flags, raise_error)


        # ------ POST-PROCESSING ------ #
        path = os.path.join(config.root, 'postprocess.py')
        flags = {
            'config': args.config,
            'dir_out': dir_out
        }
        execute_script(path, flags, raise_error)


        # ------ EVALUATION ----- #
        path = os.path.join(config.root, 'evaluate_model.py')
        flags = {
            'dir_out': dir_out,
            'grid_shape': simulation['grid_shape'],
            'plot': None
        }
        if config.benchmark:
            flags['benchmark'] = None
        execute_script(path, flags, raise_error)
