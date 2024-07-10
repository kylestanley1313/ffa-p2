import argparse
import numpy as np
import os
import time
import torch
from sklearn.decomposition import PCA
from typing import Generator

from config import load_config
from utils import (
    gen_seeds,
    multiply_list,
    gen_tensors,
    write_rows_to_csv
)



class PCALoadingInitializer(object):

    def __init__(
            self, 
            method: str, 
            num_facs: int, 
            init_prop: float, 
            generator: torch.Generator
        ) -> None:
        self.method = method
        self.num_facs = num_facs
        self.init_prop = init_prop
        self.gen = generator

    def __call__(self, dataloader: Generator) -> torch.Tensor:
        
        # Read in (possibly subsampled) data
        data = []
        n = 0
        for batch in dataloader:
            n_batch = len(batch)
            num_to_keep = int(n_batch * self.init_prop)
            idx = torch.randperm(n_batch, generator=self.gen)
            data_ = batch[idx[:num_to_keep]]
            data.append(data_)
            n += num_to_keep
        data = torch.cat(data)

        # Prepare PCA estimator
        seed = gen_seeds(self.gen, 1)
        pca = PCA(self.num_facs, svd_solver=self.method, random_state=seed)

        # Initialize loadings then write to file
        # NOTE: If X = USV^T, then covariance is
        #           C = 1/(n-1) X^TX = V(S^2 / (n-1)) V^T
        pca.fit(data)
        loads = np.matmul(
            np.diag(pca.singular_values_) / np.sqrt(n - 1),
            pca.components_
        )
        loads = torch.tensor(loads, dtype=torch.float64)

        return loads.t().contiguous()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--split', type=str, choices=['full', 'train', 'valid'])
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument('--n_facs', type=int)
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized']
    )
    parser.add_argument('--prop_init', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_bench = os.path.join(args.dir_out, 'bench')
    path_init = os.path.join(args.dir_out, f'init-loads-{args.split}.pt')
    path_other_bench = os.path.join(dir_bench, 'other-dsgd.csv')
    gen = torch.Generator().manual_seed(args.seed)


    print("Initializing loadings...")
    start = time.time()
    pca_svd_solvers = {
        'pca_full': 'full',
        'pca_arpack': 'arpack',
        'pca_randomized': 'randomized'
    }
    if args.init_method == 'random':
        n_vars = multiply_list(args.sz_space)
        loads = torch.randn(n_vars, args.n_facs, generator=gen, dtype=torch.float64)
    else:
        dataloader = gen_tensors(dir_data, f'data-time-{args.split}')
        initializer = PCALoadingInitializer(
            pca_svd_solvers[args.init_method], 
            args.n_facs, 
            args.prop_init, 
            gen
        )
        loads = initializer(dataloader)
    torch.save(loads, path_init)
    end = time.time()
    if args.benchmark:
        write_rows_to_csv(path_other_bench, [['initialization', end - start]])


    print("DONE!")

