import argparse
import os
import subprocess
import sys
from typing import Dict

from config import load_config
from utils import load_yaml, refresh_directory



def execute_script(path: str, flags: Dict[str, str]):

    # Compile arguments for subprocess.run()
    args = [sys.executable, path]
    for k, v in flags.items():
        args.append(f'--{k}')
        if v is not None:
            if isinstance(v, list):
                for item in v:
                    args.append(str(item))
            else:
                args.append(str(v))

    # Run script
    result = subprocess.run(args, capture_output=True, text=True)
    if len(result.stderr) > 0:
        raise Exception(f"Error: {result.stderr}")
    else: 
        print(result.stdout)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup simulation files
    path = os.path.join(config.root, 'setup_simulations.py')
    flags = {
        'config': args.config,
        'design': args.design
    }
    execute_script(path, flags)

    # Prepare output directories
    #   - setup_simulations.py prepares design directory
    #   - simulate_data.py prepares datasets directory
    path = os.path.join(config.root, 'out', args.design)
    path_scratch = os.path.join(config.scratch_root, 'out', args.design)
    refresh_directory(path)
    refresh_directory(path_scratch)

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
            'num_train': simulation['num_train'],
            'num_val': simulation['num_val'],
            'batch_size': int(simulation['num_train'] / 4)
        }
        execute_script(path, flags)


        # ------ ESTIMATION ----- #
        path = os.path.join(config.root, f'factor_model_{simulation["estimation"]}.py')
        flags = {
            'config': args.config,
            'dataset': sim_id,
            'dir_out': dir_out,
            'world_size': simulation['world_size'],
            'num_facs': simulation['num_facs'],
            'delta': simulation['delta'],
            'init_method': simulation['init_method'],
            'init_perc': simulation['init_perc'],
            'batch_size': simulation['batch_size'],
            'lr': simulation['lr'],
            'max_epochs': simulation['max_epochs']
        }
        execute_script(path, flags)


        # ------ EVALUATION ----- #
        path = os.path.join(config.root, 'evaluate_model.py')
        flags = {
            'dir_out': dir_out,
            'grid_shape': simulation['grid_shape'],
            'plot': None
        }
        if config.benchmark:
            flags['benchmark'] = None
        execute_script(path, flags)
