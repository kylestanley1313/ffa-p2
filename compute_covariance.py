import argparse
import os
import torch
import torch.multiprocessing as mp
from typing import List

from config import load_config
from utils import (
    flatten_dataset, 
    gen_points,
    gen_tensors,
    init_process,
    multiply_list, 
    remove_file,
    refresh_directory,
)


def compute_covariance(
        point_file: str,
        dir_cov: str,
        dir_data: str
    ) -> None:
    """For a dataset contained in `dir_data` and points contained in 
    `points_file` of `dir_cov`, computes and saves full, training, and 
    validation covariances."""

    # Read in points
    path = os.path.join(dir_cov, point_file)
    points = torch.load(path)
    rows = points[:, 0]
    cols = points[:, 1]

    def _compute_covariance(split: str) -> None: 

        # Compute covariance
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data_loader = gen_tensors(dir_data, f'data-time-{split}')
        for data in data_loader: 
            n += len(data)
            t1 = torch.sum(data[:,rows] * data[:,cols], dim=0)
            t2 = torch.sum(data[:,rows], dim=0)
            t3 = torch.sum(data[:,cols], dim=0)

        cov = (t1 - t2 * t3 / n) / (n - 1)

        # Save covariances
        cov_file = point_file.replace('points', f'cov-{split}')
        cov_path = os.path.join(dir_cov, cov_file)
        torch.save(cov, cov_path)

    _compute_covariance('full')
    _compute_covariance('train')
    _compute_covariance('valid')


def compute_covariance_for_point_files(
        rank: int,
        world_size: int,
        point_files: List[str],
        dir_cov: str,
        dir_data:str
    ) -> None:
    for point_file in point_files: 
        compute_covariance(point_file, dir_cov, dir_data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_dataset', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--prop_train_time', type=float, default=0.8)
    parser.add_argument('--prop_train_space', type=float, default=0.8)
    parser.add_argument('--bsz_time', type=int)
    parser.add_argument('--bsz_space', type=int)
    parser.add_argument('--fse', action='store_true')
    parser.add_argument('--refresh_dirs', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Set directories and paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_cov = os.path.join(args.dir_out_scratch, f'cov')
    suffix = args.dir_out.split('out/')[-1].replace('/', '_')
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')

    # Delete files from covariance directory
    if args.refresh_dirs:
        refresh_directory(dir_data)
        refresh_directory(dir_cov)

    # Flatten dataset, getting `n_time` and `sz_space` along the way
    n_time, sz_space = flatten_dataset(
        args.dir_dataset, dir_data, 
        args.bsz_time, args.bsz_space
    )
    n_vars = multiply_list(sz_space)


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


    # ---------- POINT GENERATION ---------- #
    print("Generating points...")

    n_vars = multiply_list(sz_space)
    points_loader = gen_points(sz_space, args.delta, 10*n_vars)
    n_batch = 0
    for points in points_loader: 
        path = os.path.join(dir_cov, f'points-{n_batch}.pt')
        torch.save(points, path)
        n_batch += 1
        

    # ---------- COVARIANCE COMPUTATION ---------- #
    print("Computing covariance...")

    # Divide points files among workers
    point_files = [f for f in os.listdir(dir_cov) if f.startswith('points')]
    processes = []
    remove_file(path_shared)
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process,
            args=(
                rank, args.world_size, compute_covariance_for_point_files, 
                path_shared, config.backend
            ),
            kwargs={
                'point_files': point_files[slice(rank, len(point_files), args.world_size)],
                'dir_cov': dir_cov,
                'dir_data': dir_data,
            }
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("DONE!")
