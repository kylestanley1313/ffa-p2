import argparse
import os
import torch
from itertools import product

from utils import gen_seeds, load_config, load_yaml, write_yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--superdesign', type=str)
    parser.add_argument('--split_on', type=str, nargs='+')
    args = parser.parse_args()

    config = load_config(args.config)

    # NOTE: (Splits)
    #   - Subspace Estimation: load_scheme, regime, n_facs
    #   - Subspace Expression: n_facs, path_fac_cov
    #   - Factor Score Estimation: path_mask, delta

    # Load superdesign
    path = os.path.join('superdesigns', f'{args.superdesign}.yml')
    superdesign = load_yaml(path)

    designs = []
    gen = torch.Generator().manual_seed(superdesign['base_seed'])
    for i, combo in enumerate(product(*(superdesign[field] for field in args.split_on))):
        design = superdesign.copy()
        for field, value in zip(args.split_on, combo):
            design[field] = [value]
        design['base_seed'] = gen_seeds(gen, 1)
        design_name = f'{args.superdesign}_{i+1}'
        designs.append(design_name)
        path = os.path.join(config.group_root, 'designs', f'{design_name}.yml')
        write_yaml(design, path)

    # Write superdesign.txt
    path = os.path.join('superdesigns', f'{args.superdesign}.txt')
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for design in designs:
            f.write(design + '\n')

