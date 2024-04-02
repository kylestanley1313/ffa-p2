import argparse
import os

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

    # Prepare output directories
    #   - setup_simulations.py prepares design directory
    #   - simulate_data.py prepares datasets directory
    path = os.path.join(config.root, 'out', args.design)
    path_scratch = os.path.join(config.scratch_root, 'out', args.design)
    refresh_directory(path)
    refresh_directory(path_scratch)

    # TODO: Handle (variable) seeding for different simulations

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
