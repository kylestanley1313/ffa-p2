import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from typing import Callable, List, Tuple

from config import load_config
from utils.utils import (
    flatten_dataset, 
    gen_points,
    multiply_list, 
    read_tensors,
    refresh_directory,
    remove_file,
    write_rows_to_csv
)


def fair_allocate(num_items: int, num_groups: int) -> List[int]:
    """Evenly distributes num_items across num_groups."""
    out = [num_items // num_groups] * num_groups
    remainder = num_items % num_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out


def gen_strata(nprocs: int) -> List[Tuple[Tuple]]:
    path = os.path.join('strata', f'nprocs-{nprocs}.pt')
    if os.path.exists(path):
        tensor = torch.load(path)
        num_strata = 2 * nprocs + 1
        strata = [None] * num_strata
        for s in range(num_strata):
            mask = tensor[:,0] == s
            stratum = tuple(tuple(s_) for s_ in tensor[mask, 1:].tolist())
            strata[s] = stratum
        return strata
    else:
        raise Exception(f"Strata do not exist for {nprocs} processes.")
    

def allocate_points_strat(
        rank: int,
        world_size: int,
        grid_shape: List[int], 
        delta: float, 
        dir_cov: str,
        seed: int,
    ) -> None:
    """Generates and writes the files of the form points-{rank}-{n_batch}.pt 
    and strat-{rank}-{n_batch}.pt."""    
    gen = torch.Generator().manual_seed(seed)

    # Create dict mapping this rank's blocks to their stratum
    block_map = {}
    strata = gen_strata(world_size)
    for i in range(len(strata)):
        block_map[strata[i][rank]] = i

    # Segment variables
    num_vars = multiply_list(grid_shape)
    seg_cnts = fair_allocate(num_vars, 2*world_size)
    idx = 0
    segs = torch.zeros(num_vars, dtype=torch.int32)
    for seg, cnt in enumerate(seg_cnts):
        segs[idx:(idx+cnt)] = torch.ones(cnt) * seg
        idx += cnt
    segs = segs[torch.randperm(num_vars, generator=gen)]

    # Create points and strat files for this rank
    points_loader = gen_points(grid_shape, delta, 2*num_vars)
    n_batch = 0
    for points in points_loader:

        # Get the segment of each point
        sz = len(points)
        seg0 = segs[points[:,0]]
        seg1 = segs[points[:,1]]

        # Use strat to exclude points not assigned to rank (-1) and collect
        # strata of those assigned to rank (>= 0). 
        strat = -1 * torch.ones(sz, dtype=torch.int32)
        for i in range(sz):
            block = tuple(sorted(
                [seg0[i].item(), seg1[i].item()], 
                reverse=True
            ))
            strat_ = block_map.get(block)
            if strat_ is not None:
                strat[i] = strat_
        mask = strat != -1

        # Write points and strat to file
        path_points = os.path.join(dir_cov, f'points-{rank}-{n_batch}.pt')
        path_strat = os.path.join(dir_cov, f'strat-{rank}-{n_batch}.pt')
        torch.save(points[mask], path_points)
        torch.save(strat[mask], path_strat)

        n_batch += 1


def allocate_points_ddp(
        rank: int,
        world_size: int,
        grid_shape: List[int], 
        delta: float, 
        dir_cov: str,
        seed: int,
    ) -> None:
    """Generates and writes files of the form points-{rank}-{n_batch}.pt."""

    gen = torch.Generator().manual_seed(seed)
    num_vars = multiply_list(grid_shape)
    points_loader = gen_points(grid_shape, delta, 2*num_vars)
    
    n_batch = 0
    for points in points_loader: 

        # Get rank's indices using generator common to all ranks
        start = (n_batch + rank) % world_size
        idx = torch.arange(start, len(points), world_size)
        idx = torch.randperm(len(points), generator=gen)[idx]

        # Write points to file
        path = os.path.join(dir_cov, f'points-{rank}-{n_batch}.pt')
        torch.save(points[idx], path)

        n_batch += 1


def compute_covariance(
        points_file: str,
        dir_cov: str,
        dir_data: str
    ) -> None:
    """For a dataset contained in `dir_data` and points contained in 
    `points_file` of `dir_cov`, computes and saves training and validation 
    covariances."""

    # Read in points
    path = os.path.join(dir_cov, points_file)
    points = torch.load(path)

    def _compute_covariance(split: str) -> None: 

        # Compute covariance
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data_loader = read_tensors(dir_data, f'data-{split}')
        for data in data_loader: 

            n += len(data)
            for i in range(num_points): 
                row, col = points[i]
                t1[i] += torch.sum(data[:,row] * data[:,col])
                t2[i] += torch.sum(data[:,row])
                t3[i] += torch.sum(data[:,col])

        cov = (t1 - t2 * t3 / n) / (n - 1)

        # Save covariances
        cov_file = points_file.replace('points', f'cov-{split}')
        cov_path = os.path.join(dir_cov, cov_file)
        torch.save(cov, cov_path)

    _compute_covariance('full')
    _compute_covariance('train')
    _compute_covariance('valid')


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
    parser.add_argument(
        '--world_size_est', type=int, 
        help="Number of workers used in downstream estimation."
    )
    parser.add_argument(
        '--world_size_cov', type=int, 
        help="Number of workers used in this script's covariance computation."
    )
    parser.add_argument('--delta', type=float)
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

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

    # Multiprocessing configurations
    mp.set_start_method('spawn')

    # ---------- POINT ALLOCATION ---------- #
    allocate_points_fcns = {
        'strat': allocate_points_strat,
        'ddp': allocate_points_ddp
    }
    remove_file(path_shared)
    processes = []
    for rank in range(args.world_size_est):
        p = mp.Process(
            target=init_process,
            args=(
                rank, args.world_size_est, allocate_points_fcns[args.est_method], 
                path_shared, config.backend
            ),
            kwargs={
                'grid_shape': grid_shape,
                'delta': args.delta,
                'dir_cov': dir_cov,
                'seed': args.seed,  # use same seed for all ranks
            }
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


    # ---------- DATA SPLITTING ---------- #

    gen = torch.Generator().manual_seed(args.seed)
    data_loader = read_tensors(dir_data, 'data')
    i = 0
    for data in data_loader: 
        sz = len(data)
        num_train = int(args.train_prop * sz)
        idx = torch.randperm(sz, generator=gen)
        data_train = data[idx[:num_train]]
        data_valid = data[idx[num_train:]]
        path_train = os.path.join(dir_data, f'data-train-{i}.pt')
        path_valid = os.path.join(dir_data, f'data-valid-{i}.pt')
        torch.save(data_train, path_train)
        torch.save(data_valid, path_valid)
        i += 1
        

    # ---------- COVARIANCE COMPUTATION ---------- #
    print(f"Computing covariance...")

    start = time.time()
    points_files = [f for f in os.listdir(dir_cov) if f.startswith('points')]
    pool = mp.Pool(processes=args.world_size_cov)
    results = pool.map(
        partial(compute_covariance, dir_cov=dir_cov, dir_data=dir_data), 
        points_files
    )
    pool.close()
    pool.join()
    end = time.time()
    if config.benchmark:
        write_rows_to_csv(other_bench_path, [['covariance', end - start]]) 


    # ---------- MERGE FILES ---------- #

    files = sorted(os.listdir(dir_cov))
    file_types = ['points', 'cov-full', 'cov-train', 'cov-valid']
    if args.est_method == 'strat':
        file_types.append('strat')
    for rank in range(args.world_size_est):
        for file_type in file_types:
            tensor_list = []
            for f in files:
                if f.startswith(f'{file_type}-{rank}'):
                    path = os.path.join(dir_cov, f)
                    tensor_list.append(torch.load(path))
                    remove_file(path)
            tensor = torch.cat(tensor_list)
            path = os.path.join(dir_cov, f'{file_type}-{rank}.pt')
            torch.save(tensor, path)
            