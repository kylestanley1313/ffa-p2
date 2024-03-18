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
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Optional

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils.utils import (
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

def penalty(loads: torch.Tensor, diff_mat: torch.Tensor):
    return torch.trace(loads.t() @ diff_mat @ loads)


def penalty_gradient(loads: torch.Tensor, diff_mat: torch.Tensor):
    return 2 * diff_mat @ loads


def objective(
        preds: torch.Tensor, 
        cov: torch.Tensor, 
        model: 'LowRankCovariance', 
        alpha: float, 
        diff_mat: torch.Tensor
    ):
    err = mse_loss(preds, cov)
    if alpha > 0:
        pen = penalty(model.module.loads, diff_mat)
        return err + alpha * pen
    else: 
        return err


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
        if path_init:
            self.loads = torch.load(path_init)
        else:
            self.loads = torch.randn(num_vars, num_facs, dtype=torch.float64)
        self.loads.requires_grad_()
        self.loads = nn.Parameter(self.loads)

    def forward(self, idx0, idx1):
        loads0 = self.loads[idx0]
        loads1 = self.loads[idx1]
        return (loads0 * loads1).sum(dim=1)


class DistributedCovarianceDataset(Dataset):

    def __init__(
            self, 
            dir: str, 
            rank: int, 
            world_size: int,
            prop_train: float = 0.8,
            gen: torch.Generator = torch.Generator()
        ) -> None:

        points_train_list = []
        cov_train_list = []
        points_valid_list = []
        cov_valid_list = []
        self.rank_counts = {r: 0 for r in range(world_size)}

        all_points = read_tensors(dir, 'points')
        all_cov = read_tensors(dir, 'cov')
        iter = 0
        for points, cov in zip(all_points, all_cov): 

            # Loop over ranks to aggregate counts
            for r in range(world_size):
                
                # Get all indices for rank
                start = (r + iter) % world_size  # TODO: ensure even distribution of counts
                idx = torch.arange(start, len(cov), world_size)
                
                # Set rank count
                sz = len(idx)
                num_train = int(prop_train * sz)
                self.rank_counts[r] += num_train

                if r == rank:

                    # Perform train-valid split
                    idx_ = torch.randperm(sz, generator=gen)
                    idx_train = idx[idx_[:num_train]]
                    idx_valid = idx[idx_[num_train:]]
                    
                    # Collect points/cov for rank assigned to this dataset
                    points_train_list.append(points[idx_train])
                    cov_train_list.append(cov[idx_train])
                    points_valid_list.append(points[idx_valid])
                    cov_valid_list.append(cov[idx_valid])

            iter += 1

        # Training and validation points/cov
        self.points_train = torch.row_stack(points_train_list)
        self.cov_train = torch.cat(cov_train_list)
        self.points_valid = torch.row_stack(points_valid_list)
        self.cov_valid = torch.cat(cov_valid_list)

        # Placeholder attributes set by methods
        self.points = None
        self.cov = None

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def set_training(self):
        self.points = self.points_train
        self.cov = self.cov_train

    def set_validation(self):
        self.points = self.points_valid
        self.cov = self.cov_valid
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        self.set_training()
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        self.set_validation()
        points_sz += sys.getsizeof(self.points.untyped_storage())
        cov_sz += sys.getsizeof(self.cov.untyped_storage())
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

    def set_training(self):
        self.dataset.set_training()

    def set_validation(self):
        self.dataset.set_validation()

    
# -------------------- RUN -------------------- #

def compute_covariance(points_file: str, dir_cov: str, dir_data: str) -> None:
    """For a dataset contained in `dir_data` and points contained in 
    `points_file` of `dir_cov`, computes and saves covariances."""

    # Compute covariance
    points_path = os.path.join(dir_cov, points_file)
    points = torch.load(points_path)
    num_points = len(points)
    t1 = torch.zeros(num_points, dtype=torch.float64)
    t2 = torch.zeros(num_points, dtype=torch.float64)
    t3 = torch.zeros(num_points, dtype=torch.float64)
    n = 0
    data_loader = read_tensors(dir_data, 'data')
    for data_batch in data_loader: 
        n += len(data_batch)
        for i in range(num_points): 
            row, col = points[i]
            t1[i] += torch.sum(data_batch[:,row] * data_batch[:,col])
            t2[i] += torch.sum(data_batch[:,row])
            t3[i] += torch.sum(data_batch[:,col])
    cov = (t1 - t2 * t3 / n) / (n - 1)

    # Save covariance
    cov_file = points_file.replace('points', 'cov')
    cov_path = os.path.join(dir_cov, cov_file)
    torch.save(cov, cov_path)


def process_epoch(model, dataloader, objective, optimizer):

    for points, cov in dataloader:

        # Forward pass
        preds = model(points[:,0], points[:,1])
        loss = objective(preds, cov, model)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# def sparse_allreduce_hook(
#         process_group: dist.ProcessGroup, 
#         bucket: dist.GradBucket
#     ) -> torch.futures.Future[torch.Tensor]:
#     group_to_use = process_group if process_group is not None else dist.group.WORLD

#     indices = torch.nonzero(bucket.buffer()).t()
#     values = bucket.buffer()[indices[0]]
#     sparse_tensor = torch.sparse_coo_tensor(indices, values, bucket.buffer().size())
#     # sparse_tensor = bucket.buffer().to_sparse()
#     sparse_tensor.div_(group_to_use.size())
#     fut = dist.all_reduce(  # TODO: Sparse reduction?
#         sparse_tensor, group=group_to_use, async_op=True
#     ).get_future()

#     def to_dense(fut):
#         return fut.value()[0].to_dense()
    
#     return fut.then(to_dense)


# def process_epoch_opt(model, dataloader, alpha, penalty_gradient, optimizer):

#     for points, cov in dataloader:

#         # Perform forward pass on loss
#         preds = model(points[:,0], points[:,1])
#         loss = mse_loss(preds, cov)

#         # Perform backward pass on loss function, which will allreduce a 
#         # collection of sparse gradients. Then tack on the penalty gradient
#         # (dense but common to all workers) and perform update. 
#         optimizer.zero_grad()
#         loss.backward()
#         if alpha > 0:
#             model.module.loads.grad += alpha * penalty_gradient(model.module.loads.data)
#         optimizer.step()


def compute_objective(model, dataloader, alpha, penalty, rank, world_size):

    # Compute and communicate loss
    dataset = dataloader.dataset
    preds = model(dataset.points[:,0], dataset.points[:,1])
    objective = mse_loss(preds, dataset.cov)
    if alpha > 0:
        objective += alpha * penalty(model.module.loads)
    if rank > 0: 
        dist.send(objective, 0)
    else: 
        for r in range(1, world_size):
            worker_objective = torch.zeros(1, dtype=torch.float64)
            dist.recv(worker_objective, r)
            objective += worker_objective.item()
        return objective
    
    
# def train_opt(
#         rank: int, 
#         world_size: int, 
#         config: str,
#         dir_out: str,
#         grid_shape: List[int], 
#         num_facs: int, 
#         alpha: float,
#         batch_size: int,
#         lr: float, 
#         max_epochs: int, 
#         seed: int
#     ) -> None:

#     config = load_config(config)
#     torch.set_num_threads(1)
#     gen = torch.Generator().manual_seed(seed)
    
#     # Set directories
#     dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
#     dir_bench = os.path.join(dir_out, 'bench')

#     # Get benchmarking wrappers
#     process_epoch_ = time_dist_fcn(
#         fcn=process_epoch_opt,
#         dir=dir_bench,
#         prefix='process_epoch',
#         benchmark=config.benchmark
#     )
#     Dataset_ = size_dist_obj(
#         init=DistributedCovarianceDataset,
#         dir=dir_bench,
#         prefix='dataset',
#         benchmark=config.benchmark
#     )

#     dataset = Dataset_(dir_cov, rank, world_size)
#     sampler = DistributedDatasetSampler(dataset, gen)
#     dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

#     num_vars = multiply_list(grid_shape)
#     path_init = os.path.join(dir_out, 'init_loads.pt')
#     model = LowRankCovariance(num_vars, num_facs, path_init)
#     model = DDP(model)
#     model.register_comm_hook(state=None, hook=sparse_allreduce_hook)

#     diff_mat = create_second_difference_matrix(grid_shape)
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     penalty_ = partial(penalty, diff_mat=diff_mat)
#     penalty_gradient_ = partial(penalty_gradient, diff_mat=diff_mat)

#     for epoch in range(max_epochs):

#         process_epoch_(model, dataloader, alpha, penalty_gradient_, optimizer)
#         objective = compute_objective(
#             model, dataloader, alpha, penalty_, 
#             rank, world_size
#         )
#         if rank == 0:
#             print(f"epoch = {epoch} | objective = {objective}")

#     # Save model
#     if rank == 0:
#         path = os.path.join(dir_out, 'cov-model.pth')
#         state_dict = model.state_dict()
#         state_dict['loads'] = state_dict.pop('module.loads')  # Replace DDP key
#         torch.save(state_dict, path)

#     dist.destroy_process_group()




def compute_loss(model, dataloader, objective, rank, world_size):

    def compute_loss_(dataloader):
        dataset = dataloader.dataset
        preds = model(dataset.points[:,0], dataset.points[:,1])
        loss = objective(preds, dataset.cov, model)
        if rank > 0: 
            dist.send(loss, 0)
        else: 
            for r in range(1, world_size):
                worker_loss = torch.zeros(1, dtype=torch.float64)
                dist.recv(worker_loss, r)
                loss += worker_loss.item()
            return loss

    dataloader.set_training()
    train_loss = compute_loss_(dataloader)
    dataloader.set_validation()
    valid_loss = compute_loss_(dataloader)

    return train_loss, valid_loss


def init_process(
        rank: int, 
        world_size: int, 
        path_shared: str,
        config: str,
        dir_out: str,
        grid_shape: List[int],
        num_facs: int, 
        alpha: float,
        train_prop: float,
        batch_size: int,
        lr: float, 
        patience: int,
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
        train_prop, batch_size, lr, patience, max_epochs, seed
    )


def train(
        rank: int, 
        world_size: int, 
        config: str,
        dir_out: str,
        grid_shape: List[int], 
        num_facs: int, 
        alpha: float,
        train_prop: float,
        batch_size: int,
        lr: float, 
        patience: int,
        max_epochs: int, 
        seed: int
    ) -> None:

    config = load_config(config)
    torch.set_num_threads(1)
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

    dataset = Dataset_(dir_cov, rank, world_size, train_prop, gen)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

    num_vars = multiply_list(grid_shape)
    path_init = os.path.join(dir_out, 'init_loads.pt')
    model = LowRankCovariance(num_vars, num_facs, path_init)
    model = DDP(model)

    diff_mat = create_second_difference_matrix(grid_shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    objective_ = partial(objective, alpha=alpha, diff_mat=diff_mat)

    path_model = os.path.join(dir_out, 'cov-model.pth')
    best_valid_loss = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    prev_train_loss = float('inf')
    lr = torch.tensor(lr)
    for epoch in range(max_epochs):
        
        dataloader.set_training()
        process_epoch_(model, dataloader, objective_, optimizer)
        train_loss, valid_loss = compute_loss(model, dataloader, objective_, rank, world_size)
        
        # Rank-0 worker determines whether to stop and how to update lr
        if rank == 0:
            print(f"epoch = {epoch} | train_loss = {train_loss} | valid_loss = {valid_loss}")

            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), path_model)
                epochs_waited = 0
            else:
                epochs_waited += 1
                if epochs_waited >= patience: 
                    print(f"Early stopping after {epoch + 1} epochs.")
                    early_stop = torch.tensor(True)

            # Update learning rate via bold driver
            lr *= 1.05 if train_loss < prev_train_loss else 0.5

        # Communicate early_stop and lr to non-zero ranks
        dist.barrier()
        dist.broadcast(early_stop, 0)
        if early_stop:
            break
        dist.broadcast(lr, 0)
        optimizer.param_groups[0]['lr'] = lr.item()

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
    parser.add_argument('--init_prop', type=float, default=1.0)
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    # parser.add_argument('--opt', action='store_true')
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

    # Multiprocessing configurations
    mp.set_start_method('spawn')


    # ---------- POINT GENERATION ---------- #
    cov_batch_size = 2 * num_vars
    points_loader = gen_points(grid_shape, args.delta, cov_batch_size)
    iter = 0
    for points in points_loader:
        path = os.path.join(dir_cov, f'points-{iter}.pt')
        torch.save(points, path)
        iter += 1


    # ---------- COVARIANCE PREPARATION ---------- #
    print(f"Computing covariance...")

    start = time.time()
    points_files = [f for f in os.listdir(dir_cov) if f.startswith('points')]
    pool = mp.Pool(processes=args.world_size)
    results = pool.map(
        partial(compute_covariance, dir_cov=dir_cov, dir_data=dir_data), 
        points_files
    )
    pool.close()
    pool.join()
    end = time.time()
    if config.benchmark:
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
        loads = torch.randn(args.num_facs, num_vars, generator=gen, dtype=torch.float64)
    else:

        # Read in (possibly subsampled) data
        data = []
        dataloader = read_tensors(dir_data, 'data')
        for batch in dataloader:
            n = len(batch)
            num_to_keep = int(n * args.init_prop)
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
    if config.benchmark:
        write_rows_to_csv(other_bench_path, [['initialization', end - start]])


    # ---------- DISTRIBUTED RUN ---------- #
    print("Fitting model...")

    # if args.opt: 
    #     run_fcn = train_opt
    # else: 
    #     run_fcn = train

    suffix = args.dir_out.split('/')[-1]
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')
    remove_file(path_shared)
    processes = []
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, path_shared, args.config, dir_out,
                grid_shape, args.num_facs, args.alpha,
                args.train_prop, args.batch_size, args.lr, 5, args.max_epochs, 
                seeds[rank], train, config.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

