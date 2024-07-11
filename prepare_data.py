import argparse
import os
import time
import torch
import torch.multiprocessing as mp
from functools import partial
from typing import List, Tuple

from config import load_config
from utils import (
    compute_covariance,
    flatten_dataset, 
    gen_points,
    init_process,
    multiply_list, 
    gen_tensors,
    refresh_directory,
    remove_file,
    write_rows_to_csv,
)


def fair_allocate(n_items: int, n_groups: int) -> List[int]:
    """Evenly distributes n_items across n_groups."""
    out = [n_items // n_groups] * n_groups
    remainder = n_items % n_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out


def gen_strata(nprocs: int) -> List[Tuple[Tuple]]:
    path = os.path.join('strata', f'nprocs-{nprocs}.pt')
    if os.path.exists(path):
        tensor = torch.load(path)
        n_strata = 2 * nprocs + 1
        strata = [None] * n_strata
        for s in range(n_strata):
            mask = tensor[:,0] == s
            stratum = tuple(tuple(s_) for s_ in tensor[mask, 1:].tolist())
            strata[s] = stratum
        return strata
    else:
        raise Exception(f"Strata do not exist for {nprocs} processes.")
    

def allocate_points_dssgd(
        rank: int,
        world_size: int,
        sz_space: List[int], 
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
    n_vars = multiply_list(sz_space)
    seg_cnts = fair_allocate(n_vars, 2*world_size)
    idx = 0
    segs = torch.zeros(n_vars, dtype=torch.int32)
    for seg, cnt in enumerate(seg_cnts):
        segs[idx:(idx+cnt)] = torch.ones(cnt) * seg
        idx += cnt
    segs = segs[torch.randperm(n_vars, generator=gen)]

    # Create points and strat files for this rank
    points_loader = gen_points(sz_space, delta, 2*n_vars)
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


def allocate_points_dsgd(
        rank: int,
        world_size: int,
        sz_space: List[int], 
        delta: float, 
        dir_cov: str,
        seed: int,
    ) -> None:
    """Generates and writes files of the form points-{rank}-{n_batch}.pt."""

    gen = torch.Generator().manual_seed(seed)
    n_vars = multiply_list(sz_space)
    points_loader = gen_points(sz_space, delta, 2*n_vars)
    
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


def allocate_points_lbfgs(
        sz_space: List[int], 
        delta: float, 
        dir_cov: str
    ) -> None:
    """Generates and writes file points.pt.
    
    This function will only be called for relativley small datasets, 
    so memory-efficient batching of output files need not be used. 
    """
    n_vars = multiply_list(sz_space)
    points_loader = gen_points(sz_space, delta, 2*n_vars)
    points_list = []
    for points in points_loader:
        points_list.append(points)
    points = torch.cat(points_list)
    path = os.path.join(dir_cov, 'points.pt')
    torch.save(points, path)


def compute_covariance_for_points_file(
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

    for split in ['full', 'train', 'valid']:
        cov = compute_covariance(points, dir_data, split)
        cov_file = points_file.replace('points', f'cov-{split}')
        cov_path = os.path.join(dir_cov, cov_file)
        torch.save(cov, cov_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_dataset', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--est_method', type=str, choices=['lbfgs', 'dsgd', 'dssgd'])
    parser.add_argument(
        '--world_size_est', type=int, 
        help="Number of workers used in downstream estimation."
    )
    parser.add_argument(
        '--world_size_cov', type=int, 
        help="Number of workers used in this script's covariance computation."
    )
    parser.add_argument('--delta', type=float)
    parser.add_argument('--prop_train_time', type=float, default=0.8)
    parser.add_argument('--prop_train_space', type=float, default=0.8)
    parser.add_argument('--bsz_time', type=int)
    parser.add_argument('--bsz_space', type=int)
    parser.add_argument('--fse', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--refresh_dirs', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Set directories and paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_cov = os.path.join(args.dir_out_scratch, f'cov-{args.est_method}')
    dir_bench = os.path.join(args.dir_out, 'bench')
    other_bench_path = os.path.join(dir_bench, f'other-{args.est_method}.csv')
    suffix = args.dir_out.split('out/')[-1].replace('/', '_')
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')

    # Delete files from various directories
    if args.refresh_dirs:
        refresh_directory(dir_data)
        refresh_directory(dir_cov)
        if args.benchmark:
            refresh_directory(dir_bench)

    # Flatten dataset, getting `n_time` and `sz_space` along the way
    n_time, sz_space = flatten_dataset(
        args.dir_dataset, dir_data, 
        args.bsz_time, args.bsz_space
    )
    n_vars = multiply_list(sz_space)


    # Multiprocessing configurations
    mp.set_start_method('spawn')

    # ---------- POINT ALLOCATION ---------- #
    print("Allocating points...")
    allocate_points_fcns = {
        'lbfgs': allocate_points_lbfgs,
        'dsgd': allocate_points_dsgd,
        'dssgd': allocate_points_dssgd,
    }

    if args.est_method in ['dsgd', 'dssgd']:  # Distributed estimation methods
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
                    'sz_space': sz_space,
                    'delta': args.delta,
                    'dir_cov': dir_cov,
                    'seed': args.seed,  # use same seed for all ranks
                }
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

    else:  # Serial estimation methods
        allocate_points_fcns[args.est_method](sz_space, args.delta, dir_cov)



    # ---------- DATA SPLITTING ---------- #
    gen = torch.Generator().manual_seed(args.seed)

    # NOTE: Loading and error covariance estimation is made easy by splitting 
    # data directly. Factor estimation is made easy by indirectly splitting the
    # data via the flattened spatial indices. 
        
    print("Splitting data on time...")
    data_loader = gen_tensors(dir_data, 'data-time-full')
    i = 0
    for data in data_loader: 
        sz = len(data)
        n_train = int(args.prop_train_time * sz)
        idx = torch.randperm(sz, generator=gen)
        data_train = data[idx[:n_train]]
        data_valid = data[idx[n_train:]]
        path_train = os.path.join(dir_data, f'data-time-train-{i}.pt')
        path_valid = os.path.join(dir_data, f'data-time-valid-{i}.pt')
        torch.save(data_train, path_train)
        torch.save(data_valid, path_valid)
        i += 1

    if args.fse:
        print("Splitting indices on space...")
        idx = torch.randperm(n_vars, generator=gen)
        n_train = int(args.prop_train_space * n_vars)
        path_train = os.path.join(dir_data, f'idx-space-train.pt')
        path_valid = os.path.join(dir_data, f'idx-space-valid.pt')
        torch.save(idx[:n_train].sort().values, path_train)
        torch.save(idx[n_train:].sort().values, path_valid)
        

    # ---------- COVARIANCE COMPUTATION ---------- #
    print("Computing covariance...")

    start = time.time()
    points_files = [f for f in os.listdir(dir_cov) if f.startswith('points')]
    pool = mp.Pool(processes=args.world_size_cov)
    results = pool.map(
        partial(compute_covariance_for_points_file, dir_cov=dir_cov, dir_data=dir_data), 
        points_files
    )
    pool.close()
    pool.join()
    end = time.time()
    if args.benchmark:
        write_rows_to_csv(other_bench_path, [['covariance', end - start]]) 


    # ---------- MERGE FILES ---------- #
    print("Merging files...")

    if args.est_method in ['dsgd', 'dssgd']:
        files = sorted(os.listdir(dir_cov))
        file_types = ['points', 'cov-full', 'cov-train', 'cov-valid']
        if args.est_method == 'dssgd':
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

            