import argparse
import itertools
import torch
import os

from config import load_config
from utils.utils import gen_seeds, load_yaml, refresh_directory, write_yaml


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    args = parser.parse_args()

    # Load config and design
    config = load_config(args.config)
    design = load_yaml(os.path.join(config.root, 'designs', f'{args.design}.yml'))
    
    # Define globals
    num_reps = design.pop('num_reps')
    base_seed = design.pop('base_seed')
    gen = torch.Generator().manual_seed(base_seed)
    
    # Prepare design directory
    dir_design = os.path.join(config.root, 'designs', args.design)
    refresh_directory(dir_design)

    # Create simulation directories and write repetition YAMLs
    fields = list(design.keys())
    idx_list = [list(range(len(design[f]))) for f in fields]
    sim_cnt = 0
    for idx in itertools.product(*idx_list):

        # Create simulation directory
        simulation = {fields[i]: design[fields[i]][idx[i]] for i in range(len(fields))}
        dir_sim = os.path.join(dir_design, f'sim-{sim_cnt}')
        os.mkdir(dir_sim)
        rep_seeds = gen_seeds(gen, num_reps)
        if num_reps == 1: 
            rep_seeds = [rep_seeds]
        
        # Write repetition YAMLs
        for r in range(num_reps):
            repetition = simulation.copy()
            repetition['seed'] = rep_seeds[r]
            path = os.path.join(dir_sim, f'rep-{r}.yml')
            write_yaml(repetition, path)

        sim_cnt += 1

