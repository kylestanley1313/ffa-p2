import argparse
import os
import numpy as np
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from functools import partial
from sklearn.decomposition import PCA
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Optional

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils import (
    create_second_difference_matrix, 
    flatten_dataset,
    gen_points, 
    gen_seeds, 
    multiply_list,
    read_tensors,
    refresh_directory,
    remove_file,
    write_rows_to_csv
)



# -------------------- UTILITIES -------------------- #

def roughness_penalty(loads: torch.Tensor, diff_mat: torch.Tensor):
    return torch.trace(loads.t() @ diff_mat @ loads)


def objective(
        preds: torch.Tensor, 
        cov: torch.Tensor
    ):
    return mse_loss(preds, cov)


# -------------------- MODULES -------------------- #

class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            num_vars: int, 
            num_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.num_facs = num_facs
        self.loads = nn.Embedding(num_vars, num_facs, dtype=torch.float64)
        if path_init:
            self.loads.weight.data = torch.load(path_init)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)


class DistributedCovarianceDataset(Dataset):

    def __init__(self, dir: str, rank: int, world_size: int):
        points_list = []
        cov_list = []
        self.rank_counts = {r: 0 for r in range(world_size)}
        all_points = read_tensors(dir, 'points')
        all_cov = read_tensors(dir, 'cov')
        iter = 0
        for points, cov in zip(all_points, all_cov): 

            # Loop over ranks to aggregate counts
            for r in range(world_size):
                start = (r + iter) % world_size  # TODO: ensure even distribution of counts
                idx = torch.arange(start, len(cov), world_size)
                self.rank_counts[r] += len(idx)

                # Collect points/cov for rank assigned to this dataset
                if r == rank:
                    points_list.append(points[idx])
                    cov_list.append(cov[idx])

            iter += 1

        self.points = torch.row_stack(points_list)
        self.cov = torch.cat(cov_list)

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        return points_sz + cov_sz
    

class DistributedDatasetSampler(Sampler):

    def __init__(self, dataset, gen):
        self.dataset = dataset
        self.gen = gen
        self.num_iters = max(dataset.rank_counts.values())

    def __iter__(self):
        idx = torch.randperm(len(self.dataset), generator=self.gen)
        pad_size = self.num_iters - len(idx)
        idx_pad = torch.randperm(len(idx), generator=self.gen)[:pad_size]
        return iter(idx.tolist() + idx_pad.tolist())
    

class BasicDataLoader(object):

    def __init__(self, dataset, sampler, batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        curr_batch = torch.zeros(self.batch_size, dtype=torch.int32)
        cnt = 0
        for idx in self.sampler:
            curr_batch[cnt] = idx
            cnt += 1
            if cnt == self.batch_size:
                yield self.dataset[curr_batch]
                curr_batch = torch.zeros(self.batch_size, dtype=torch.int32)
                cnt = 0

        # Yield the last batch if it's not a complete batch
        if cnt > 0:
            yield self.dataset[curr_batch[:cnt]]

    
# -------------------- RUN -------------------- #


def process_epoch(model, dataloader, objective, optimizer):

    bench = True if dist.get_rank() == 0 else False
    time_model = 0
    time_objective = 0
    time_backward = 0
    time_step = 0

    for points, cov in dataloader:

        # Forward pass
        start = time.time()
        preds = model(points[:,0], points[:,1])
        end = time.time()
        time_model += end - start
        start = time.time()
        loss = objective(preds, cov)
        end = time.time()
        time_objective += end - start

        # Backward pass
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        end = time.time()
        time_backward += end - start
        start = time.time()
        optimizer.step()
        end = time.time()
        time_step += end - start
    
    msg = (
        f"""
        Times: 
            time_model = {time_model}
            time_objective = {time_objective}
            time_backward = {time_backward}
            time_step = {time_step}
        """
    )
    if bench: 
        print(msg)



def compute_loss(model, dataloader, objective, rank, world_size):

    # Compute and communicate loss
    dataset = dataloader.dataset
    preds = model(dataset.points[:,0], dataset.points[:,1])
    loss = objective(preds, dataset.cov)
    if rank > 0: 
        dist.send(loss, 0)
    else: 
        for r in range(1, world_size):
            worker_loss = torch.zeros(1, dtype=torch.float64)
            dist.recv(worker_loss, r)
            loss += worker_loss.item()
        return loss


def init_process(
        rank: int, 
        world_size: int, 
        path_shared: str,
        config: str,
        dir_out: str,
        grid_shape: List[int],
        num_facs: int, 
        alpha: float,
        batch_size: int,
        lr: float, 
        max_epochs: int, 
        seed: int, 
        fcn: Callable, 
        backend: str
    ):
    dist.init_process_group(
        backend, init_method=f'file://{path_shared}',
        rank=rank, world_size=world_size
    )
    fcn(
        rank, world_size, config, dir_out,
        grid_shape, num_facs, alpha,
        batch_size, lr, max_epochs, seed
    )


def run(
        rank: int, 
        world_size: int, 
        config: str,
        dir_out: str,
        grid_shape: List[int], 
        num_facs: int, 
        alpha: float,
        batch_size: int,
        lr: float, 
        max_epochs: int, 
        seed: int
    ) -> None:

    config = load_config(config)
    gen = torch.Generator().manual_seed(seed)
    
    # Set directories
    dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
    dir_bench = os.path.join(dir_out, 'bench')

    # Get benchmarking wrappers
    process_epoch_ = time_dist_fcn(
        fcn=process_epoch,
        dir=dir_bench,
        prefix='process_epoch',
        benchmark=config.benchmark
    )
    Dataset_ = size_dist_obj(
        init=DistributedCovarianceDataset,
        dir=dir_bench,
        prefix='dataset',
        benchmark=config.benchmark
    )

    dataset = Dataset_(dir_cov, rank, world_size)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

    num_vars = multiply_list(grid_shape)
    path_init = os.path.join(dir_out, 'init_loads.pt')
    model = LowRankCovariance(num_vars, num_facs, path_init)
    model = DDP(model)

    diff_mat = create_second_difference_matrix(grid_shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(max_epochs):

        process_epoch_(model, dataloader, objective, optimizer)
        loss = compute_loss(model, dataloader, objective, rank, world_size)
        if rank == 0:
            print(f"epoch = {epoch} | loss = {loss}")

    # Save model
    if rank == 0:
        path = os.path.join(dir_out, 'cov-model.pth')
        state_dict = model.state_dict()
        state_dict['loads.weight'] = state_dict.pop('module.loads.weight')  # Replace DDP key
        torch.save(state_dict, path)

    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--num_facs', type=int)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--delta', type=float)
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized']
    )
    parser.add_argument('--init_perc', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set directories and paths
    dir_dataset = os.path.join(config.scratch_root, 'datasets', args.dataset)
    dir_out = os.path.join('out', args.dir_out)
    dir_data = os.path.join(config.scratch_root, dir_out, 'data')
    dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
    dir_bench = os.path.join(dir_out, 'bench')
    path_init = os.path.join(dir_out, 'init_loads.pt')
    path_model = os.path.join(dir_out, 'cov-model.pth')
    other_bench_path = os.path.join(dir_bench, 'other.csv')
    
    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Delete files from various directories
    refresh_directory(dir_data)
    refresh_directory(dir_cov)
    if config.benchmark:
        refresh_directory(dir_bench)
    remove_file(path_init)
    remove_file(path_model)

    # Flatten dataset, getting `grid_shape` and `num_vars` along the way
    grid_shape = flatten_dataset(dir_dataset, dir_data)
    num_vars = multiply_list(grid_shape)


    # ---------- COVARIANCE PREPARATION ---------- $
    print(f"Computing covariance...")

    # Generate points then compute covariance
    start = time.time()
    cov_batch_size_per_proc = int((num_vars ** 2) / 18)  # TODO: Why? Better choice?
    cov_batch_size = cov_batch_size_per_proc * args.world_size
    points = gen_points(grid_shape, args.delta, cov_batch_size)
    iter = 0
    for points_batch in points: 
        num_points = len(points_batch)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data = read_tensors(dir_data, 'data')
        for data_batch in data: 
            n += len(data_batch)
            for i in range(num_points): 
                row, col = points_batch[i]
                t1[i] += torch.sum(data_batch[:,row] * data_batch[:,col])
                t2[i] += torch.sum(data_batch[:,row])
                t3[i] += torch.sum(data_batch[:,col])
        cov = (t1 - t2 * t3 / n) / (n - 1)
        path_points = os.path.join(dir_cov, f'points-{iter}.pt')
        path_cov = os.path.join(dir_cov, f'cov-{iter}.pt')
        torch.save(points_batch, path_points)
        torch.save(cov, path_cov)
        iter += 1
    end = time.time()
    write_rows_to_csv(other_bench_path, [['covariance', end - start]])
        

    # ---------- INITIALIZATION ---------- #
    print("Initializing loadings...")

    start = time.time()
    pca_svd_solvers = {
        'pca_full': 'full',
        'pca_arpack': 'arpack',
        'pca_randomized': 'randomized'
    }
    if args.init_method == 'random':
        init_loads = torch.randn(num_vars, args.num_facs, generator=gen, dtype=torch.float64)
    else:

        # Read in (possibly subsampled) data
        data = []
        dataloader = read_tensors(dir_data, 'data')
        for batch in dataloader:
            n = len(batch)
            num_to_keep = int(n * args.init_perc)
            idx = torch.randperm(n, generator=gen)
            data_ = batch[idx[:num_to_keep]]
            data.append(data_)
        data = torch.cat(data)

        # Prepare PCA estimator
        seed = gen_seeds(gen, 1)
        svd_solver = pca_svd_solvers[args.init_method]
        pca = PCA(args.num_facs, svd_solver=svd_solver, random_state=seed)

        # Initialize loadings then write to file
        pca.fit(data)
        loads = np.matmul(
            np.diag(np.sqrt(pca.singular_values_)),
            pca.components_
        )
        loads = torch.tensor(loads, dtype=torch.float64)
        torch.save(loads.t(), path_init)
    end = time.time()
    write_rows_to_csv(other_bench_path, [['initialization', end - start]])


    # ---------- DISTRIBUTED RUN ---------- #
    print("Fitting model...")

    suffix = args.dir_out.split('/')[-1]
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')
    remove_file(path_shared)
    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, path_shared, args.config, dir_out,
                grid_shape, args.num_facs, args.alpha,
                args.batch_size, args.lr, args.max_epochs, 
                seeds[rank], run, config.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
