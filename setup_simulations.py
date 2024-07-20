import argparse
import itertools
import torch
import os

from config import load_config
from utils import gen_seeds, load_yaml, refresh_directory, write_yaml


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--fse', action='store_true')
    args = parser.parse_args()

    # Load config and design
    config = load_config(args.config)
    design = load_yaml(os.path.join(config.root, 'designs', f'{args.design}.yml'))
    
    # Define globals
    n_reps = design.pop('n_reps')
    base_seed = design.pop('base_seed')
    gen = torch.Generator().manual_seed(base_seed)
    
    # Prepare design directories
    dir_design = os.path.join(config.root, 'designs', args.design)
    dir_design_dataset = os.path.join(config.scratch_root, 'datasets', args.design)
    dir_design_out = os.path.join(config.root, 'out', args.design)
    dir_design_out_scratch = os.path.join(config.scratch_root, 'out', args.design)
    refresh_directory(dir_design)
    refresh_directory(dir_design_dataset)
    refresh_directory(dir_design_out)
    refresh_directory(dir_design_out_scratch)

    # Create simulation directories and write repetition YAMLs
    fields = list(design.keys())
    idx_list = [list(range(len(design[f]))) for f in fields]
    sim_cnt = 0
    for idx in itertools.product(*idx_list):

        # Create simulation directory
        sim = {fields[i]: design[fields[i]][idx[i]] for i in range(len(fields))}
        dir_sim = os.path.join(dir_design, f'sim-{sim_cnt}')
        os.mkdir(dir_sim)
        rep_seeds = gen_seeds(gen, n_reps)
        if n_reps == 1: 
            rep_seeds = [rep_seeds]

        # Create directories, then write YAML
        for r in range(n_reps):

            rep = sim.copy()
            rep['dir_dataset'] = os.path.join(
                config.scratch_root, 'datasets', 
                args.design, f'sim-{sim_cnt}', f'rep-{r}'
            )
            rep['dir_out_sim'] = os.path.join(
                config.root, 'out', 
                args.design, f'sim-{sim_cnt}'
            )
            rep['dir_out'] = os.path.join(rep['dir_out_sim'], f'rep-{r}')
            rep['dir_out_scratch'] = os.path.join(
                config.scratch_root, 'out', 
                args.design, f'sim-{sim_cnt}', f'rep-{r}'
            )
            os.makedirs(rep['dir_dataset'])
            os.makedirs(rep['dir_out'])
            if not os.path.exists(rep['dir_out_scratch']):
                os.makedirs(rep['dir_out_scratch'])
            os.makedirs(os.path.join(rep['dir_out_scratch'], 'data'))
            os.makedirs(os.path.join(rep['dir_out_scratch'], 'cov'))
            if args.fse:
                os.makedirs(os.path.join(rep['dir_out'], 'err-cov'))
            for m in ['lbfgs', 'dsgd', 'dssgd']: 
                os.makedirs(os.path.join(rep['dir_out_scratch'], f'idx-{m}'))
            if args.benchmark: 
                os.makedirs(os.path.join(rep['dir_out'], 'bench'))
            os.makedirs(os.path.join(rep['dir_out'], 'results'))

            rep['seed'] = rep_seeds[r]
            path = os.path.join(dir_sim, f'rep-{r}.yml')
            write_yaml(rep, path)

        sim_cnt += 1

