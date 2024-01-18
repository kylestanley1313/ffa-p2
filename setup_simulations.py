import argparse
import itertools
import os

from config import load_config
from utils import load_yaml, refresh_directory, write_yaml


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    args = parser.parse_args()

    # Load config and design
    config = load_config(args.config)
    design = load_yaml(os.path.join(config.root, 'designs', f'{args.design}.yml'))

    # Prepare design directory
    dir_design = os.path.join(config.root, 'designs', args.design)
    refresh_directory(dir_design)

    # Write simulations to design directory
    fields = list(design.keys())
    idx_list = [list(range(len(design[f]))) for f in fields]
    cnt = 0
    for idx in itertools.product(*idx_list):
        simulation = {fields[i]: design[fields[i]][idx[i]] for i in range(len(fields))}
        path = os.path.join(dir_design, f'{args.design}_{cnt}.yml')
        write_yaml(simulation, path)
        cnt += 1

