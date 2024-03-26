import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, List

from config import load_config
from utils.utils import (
    flatten_dataset, 
    gen_points,
    gen_seeds, 
    multiply_list, 
    refresh_directory,
    remove_file
)


def allocate_points_strat():
    pass


def allocate_points_ddp(
        rank: int,
        world_size: int,
        grid_shape: List[int], 
        delta: float, 
        prop_train: float,
        dir_cov: str,
        seed: int = 12345
    ) -> None:
    gen = torch.Generator().manual_seed(seed)
    gen_rank = torch.Generator().manaul_seed(seed + rank)

    num_vars = multiply_list(grid_shape)
    points_loader = gen_points(grid_shape, delta, 2*num_vars)
    
    iter = 0
    points_train = []
    points_valid = []
    for points in points_loader: 

        # Get rank's indices using generator common to all ranks
        start = (iter + rank) % world_size
        idx = torch.arange(start, len(points), world_size)
        idx = torch.randperm(len(points), generator=gen)[idx]

        # Perform train-valid split using generator unique to this rank
        sz = len(idx)
        num_train = int(prop_train * sz)
        idx_ = torch.randperm(sz, generator=gen_rank)
        idx_train = idx[idx_[:num_train]]
        idx_valid = idx[idx_[num_train:]]

        # Collect points
        points_train.append(points[idx_train])
        points_valid.append(points[idx_valid])

        iter += 1
    
    points_train = torch.row_stack(points_train)
    points_valid = torch.row_stack(points_valid)

    path_train = os.path.join(dir_cov, f'points-{rank}-train.pt')
    path_valid = os.path.join(dir_cov, f'points-{rank}-valid.pt')
    torch.save(points_train, path_train)
    torch.save(points_valid, path_valid)


def stratify_points():
    pass


def compute_covariance():
    pass


def init_process(
        rank: int, 
        world_size: int, 
        fcn: Callable, 
        path_shared: str,
        backend: str,
        **kwargs
    ):
    dist.init_process_group(
        backend, init_method=f'file://{path_shared}',
        rank=rank, world_size=world_size
    )
    fcn(rank, world_size, **kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--est_method', type=str, choices=['strat', 'ddp'])
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Configure globals
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Set directories and paths
    dir_dataset = os.path.join(config.scratch_root, 'datasets', args.dataset)
    dir_out = os.path.join('out', args.dir_out)
    dir_data = os.path.join(config.scratch_root, dir_out, 'data')
    dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
    dir_bench = os.path.join(dir_out, 'bench')
    other_bench_path = os.path.join(dir_bench, 'other.csv')
    suffix = args.dir_out.split('/')[-1]
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')

    # Delete files from various directories
    refresh_directory(dir_data)
    refresh_directory(dir_cov)
    if config.benchmark:
        refresh_directory(dir_bench)

    # Flatten dataset, getting `grid_shape` and `num_vars` along the way
    grid_shape = flatten_dataset(dir_dataset, dir_data)
    # num_vars = multiply_list(grid_shape)

    # Multiprocessing configurations
    mp.set_start_method('spawn')

    # ---------- POINT ALLOCATION ---------- #
    if args.est_method == 'strat':
        pass
        # out: points_rank_train, points_rank_val, strat_rank_train
    
    elif args.est_method == 'ddp':

        remove_file(path_shared)
        processes = []
        for rank in range(args.world_size):
            p = mp.Process(
                target=init_process,
                args=(
                    rank, args.world_size, allocate_points_ddp, 
                    path_shared, config.backend
                ),
                kwargs={
                    'grid_shape': grid_shape,
                    'delta': args.delta,
                    'dir_cov': dir_cov,
                    'seed': args.seed  # same seed for each rank
                }
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()



        # out: points_rank_train and points_rank_val

    else: 
        raise Exception("Invalid argument passed to `est_method`!")


