import argparse
import numpy as np
import os
import torch
import torch.multiprocessing as mp
from typing import List, Optional

from config import load_config
from utils import (
    flatten_dataset, 
    gen_points,
    gen_tensors,
    get_field_from_fname,
    init_process,
    multiply_list, 
    read_tensors,
    remove_file,
    refresh_directory,
)


def assign_spatial_folds(
        sz_space: List[int], 
        n_folds_by_dim: List, 
        path_mask: Optional[str] = None
    ) -> torch.Tensor: 
    """Creates folds for spatial cross-validation by "folding" along axes.

    Args:
        sz_space (List[int]): Shape of spatial tensor.
        n_folds_by_dim (List): Number of folds along each dimension.
        path_mask (Optional[str]): Path to mask tensor.

    Raises:
        Exception: If number of dimensions is not 1, 2, or 3.

    Returns:
        torch.Tensor: A 1-D fold-assignment tensor.
    """
    ndim = len(sz_space)

    # Get breaks
    breaks = []
    for d in range(ndim):
        breaks.append(
            torch.linspace(
                0, sz_space[d], n_folds_by_dim[d] + 1, 
                dtype=torch.int32
            ).tolist()
        )
    folds = torch.full(sz_space, float('nan'))

    if ndim == 1: 
        for v in range(n_folds_by_dim[0]):
            min_, max_ = breaks[0][v:(v+2)]
            folds[min_:max_] = v

    elif ndim == 2: 
        v = 0
        for v1 in range(n_folds_by_dim[0]):
            min_1, max_1 = breaks[0][v1:(v1+2)]

            for v2 in range(n_folds_by_dim[1]):
                min_2, max_2 = breaks[1][v2:(v2+2)]
                folds[min_1:max_1, min_2:max_2] = v
                v += 1

    elif ndim == 3: 
        v = 0
        for v1 in range(n_folds_by_dim[0]):
            min_1, max_1 = breaks[0][v1:(v1+2)]

            for v2 in range(n_folds_by_dim[1]):
                min_2, max_2 = breaks[1][v2:(v2+2)]

                for v3 in range(n_folds_by_dim[2]):
                    min_3, max_3 = breaks[2][v3:(v3+2)]
                    folds[min_1:max_1, min_2:max_2, min_3:max_3] = v
                    v += 1

    else: 
        raise Exception("Support not provided for spatial dimensions greater than 3!")
    
    # Apply mask
    folds = torch.flatten(folds)
    if path_mask: 
        mask = torch.flatten(torch.load(path_mask))
        folds = folds[mask]

    return folds


def compute_suff_stats(
        point_file: str,
        dir_cov: str,
        dir_data: str,
        fold: int,
    ) -> None:

    # Read in points
    i = get_field_from_fname(point_file, 'i')
    path = os.path.join(dir_cov, point_file)
    points = torch.load(path)
    rows = points[:,0]
    cols = points[:,1]

    # Prepare suff stat paths
    path_1 = os.path.join(dir_cov, f't1_v-{fold}_i-{i}_.pt')
    path_2 = os.path.join(dir_cov, f't2_v-{fold}_i-{i}_.pt')
    path_3 = os.path.join(dir_cov, f't3_v-{fold}_i-{i}_.pt')

    # Compute scaled covariance
    num_points = len(points)
    t1 = torch.zeros(num_points, dtype=torch.float32)
    t2 = torch.zeros(num_points, dtype=torch.float32)
    t3 = torch.zeros(num_points, dtype=torch.float32)
    data_loader = gen_tensors(dir_data, f'data-time_v-{fold}_')
    for data in data_loader: 
        t1 += torch.sum(data[:,rows] * data[:,cols], dim=0)
        t2 += torch.sum(data[:,rows], dim=0)
        t3 += torch.sum(data[:,cols], dim=0)
    torch.save(t1, path_1)
    torch.save(t2, path_2)
    torch.save(t3, path_3)


def compute_suff_stats_for_point_files(
        rank: int,
        world_size: int,
        point_files: List[str],
        dir_cov: str,
        dir_data:str,
        n_folds: int
    ) -> None:
    for point_file in point_files: 
        for fold in range(n_folds): 
            compute_suff_stats(point_file, dir_cov, dir_data, fold)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_dataset', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--n_folds_sub', type=int)
    parser.add_argument('--n_folds_space_by_dim', type=int, nargs='+')
    parser.add_argument('--n_sub', type=int)
    parser.add_argument('--bsz_time', type=int)
    parser.add_argument('--bsz_space', type=int)
    parser.add_argument('--path_mask', type=str)
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
    for n in range(args.n_sub):
        n_time, sz_space = flatten_dataset(
            args.dir_dataset, dir_data, n,
            args.bsz_time, args.bsz_space,
            args.path_mask
        )

    # Get the number of spatial locations
    if args.path_mask is None: 
        n_vars = multiply_list(sz_space)
    else: 
        n_vars = torch.nonzero(torch.load(args.path_mask)).shape[0]

    # Multiprocessing settings
    mp.set_start_method('spawn')


    # ---------- DATA SPLITTING ---------- #
    gen = torch.Generator().manual_seed(args.seed)

    # NOTE: 
    #   - For loading estimation, split on subjects. 
    #   - For factor score estimation, indirectly split on space via the
    #     flattened spatial indices.

    print("Splitting data on subjects...")

    # Partition subjects into folds
    subs = torch.randperm(args.n_sub, generator=gen)
    sizes = torch.full((args.n_folds_sub,), args.n_sub // args.n_folds_sub)
    sizes[:args.n_sub % args.n_folds_sub] += 1
    folds = torch.split(subs, sizes.tolist()) 

    # Create folds
    sz_folds = torch.zeros(args.n_folds_sub)
    for v, fold in enumerate(folds): 

        # Save fold
        path = os.path.join(dir_data, f'subs_v-{v}.pt')
        torch.save(fold, path)

        sz_fold = 0
        for i, n in enumerate(fold.tolist()):

            # Read data and extract size
            data = read_tensors(dir_data, f'data-time_split-full_n-{n}_', sort_by=('i', int))
            sz_fold += len(data)

            # Save data
            path = os.path.join(dir_data, f'data-time_v-{v}_i-{i}_.pt')
            torch.save(data, path)

        sz_folds[v] = sz_fold

    # Save fold sizes
    path = os.path.join(dir_cov, 'sz_folds.pt')
    torch.save(sz_folds, path)

    if args.fse: 
        print("Splitting data on space...")
        n_folds_space = multiply_list(args.n_folds_space_by_dim)
        folds = assign_spatial_folds(sz_space, args.n_folds_space_by_dim, args.path_mask)
        for v in range(n_folds_space):
            idx_train = torch.where(folds != v)[0]
            idx_valid = torch.where(folds == v)[0]
            path_train = os.path.join(dir_data, f'idx-space_split-train_v-{v}_.pt')
            path_valid = os.path.join(dir_data, f'idx-space_split-valid_v-{v}_.pt')
            torch.save(idx_train, path_train)
            torch.save(idx_valid, path_valid)


    # ---------- POINT GENERATION ---------- #
    print("Generating points...")

    points_loader = gen_points(sz_space, args.delta, 100*n_vars, path_mask=args.path_mask)
    n_batch = 0
    for points in points_loader:
        path = os.path.join(dir_cov, f'points_i-{n_batch}_.pt')
        torch.save(points, path)
        n_batch += 1
        

    # ---------- SUFF STAT COMPUTATION ---------- #
    print("Computing suff stats...")

    processes = []
    remove_file(path_shared)
    point_files = [f for f in os.listdir(dir_cov) if f.startswith('points_')]
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process,
            args=(
                rank, args.world_size, compute_suff_stats_for_point_files, 
                path_shared, config.backend
            ),
            kwargs={
                'point_files': point_files[slice(rank, len(point_files), args.world_size)],
                'dir_cov': dir_cov,
                'dir_data': dir_data,
                'n_folds': args.n_folds_sub,
            }
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    # ---------- SUFF STAT AGGREGATION ---------- #

    print("Aggregating suff stats...")

    sz_folds = torch.load(os.path.join(dir_cov, 'sz_folds.pt'))
    all_folds = torch.arange(args.n_folds_sub)
    for i in range(n_batch):

        # Load suff stats for batch
        t1_folds = []
        t2_folds = []
        t3_folds = []
        for v in range(args.n_folds_sub):
            t1_folds.append(torch.load(os.path.join(dir_cov, f't1_v-{v}_i-{i}_.pt')))
            t2_folds.append(torch.load(os.path.join(dir_cov, f't2_v-{v}_i-{i}_.pt')))
            t3_folds.append(torch.load(os.path.join(dir_cov, f't3_v-{v}_i-{i}_.pt')))
        t1_folds = torch.vstack(t1_folds)
        t2_folds = torch.vstack(t2_folds)
        t3_folds = torch.vstack(t3_folds)

        # Full covariance
        sz = torch.sum(sz_folds)
        t1 = torch.sum(t1_folds, dim=0)
        t2 = torch.sum(t2_folds, dim=0)
        t3 = torch.sum(t3_folds, dim=0)
        cov = (t1 - t2 * t3 / sz) / (sz - 1)
        path = os.path.join(dir_cov, f'cov_split-full_i-{i}_.pt')
        torch.save(cov, path)

        # Folds
        for v in range(args.n_folds_sub):

            # Validation covariance
            sz = sz_folds[v]
            t1 = t1_folds[v]
            t2 = t2_folds[v]
            t3 = t3_folds[v]
            cov = (t1 - t2 * t3 / sz) / (sz - 1)
            path = os.path.join(dir_cov, f'cov_split-valid_v-{v}_i-{i}_.pt')
            torch.save(cov, path)

            # Training covariance
            sz = torch.sum(sz_folds[all_folds != v])
            t1 = torch.sum(t1_folds[all_folds != v], dim=0)
            t2 = torch.sum(t2_folds[all_folds != v], dim=0)
            t3 = torch.sum(t3_folds[all_folds != v], dim=0)
            cov = (t1 - t2 * t3 / sz) / (sz - 1)
            path = os.path.join(dir_cov, f'cov_split-train_v-{v}_i-{i}_.pt')
            torch.save(cov, path)

    print("DONE!")
