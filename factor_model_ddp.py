import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from functools import partial
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Optional

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils.initialization import PCALoadingInitializer
from utils.utils import (
    create_second_difference_matrix, 
    gen_seeds, 
    multiply_list,
    read_tensors,
    remove_file,
    write_rows_to_csv
)



# -------------------- UTILITIES -------------------- #

def loss_fcn(preds, cov, num_vars):
    return torch.sum((preds - cov) ** 2) / num_vars ** 2


def penalty_fcn(loads: torch.Tensor, alpha: float, diff_mat: torch.Tensor):
    num_vars, num_facs = loads.shape
    return alpha * torch.trace(loads.t() @ diff_mat @ loads) / num_vars / num_facs


def penalty_fcn_gradient(loads: torch.Tensor, diff_mat: torch.Tensor):
    num_vars, num_facs = loads.shape
    return 2 * diff_mat @ loads / num_vars / num_facs


# def objective(
#         preds: torch.Tensor, 
#         cov: torch.Tensor, 
#         model: 'LowRankCovariance', 
#         alpha: float, 
#         diff_mat: torch.Tensor
#     ):
#     err = torch.sum((preds - cov) ** 2) / model.module.num_vars ** 2
#     if alpha > 0:
#         pen = penalty(model.module.loads, diff_mat)
#         return err + alpha * pen
#     else: 
#         return err


# -------------------- MODULES -------------------- #

class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            num_vars: int, 
            num_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.num_vars = num_vars
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
            world_size: int
        ) -> None:

        # Get point counts for all ranks and save points for this rank
        self.rank_counts = {}
        for r in range(world_size):
            path = os.path.join(dir, f'points-{rank}.pt')
            points = torch.load(path)
            self.rank_counts[r] = len(points)
            if rank == r:
                self.points = points

        # Get train/valid covariance for this rank
        path = os.path.join(dir, f'cov-train-{rank}.pt')
        self.cov_train = torch.load(path)
        path = os.path.join(dir, f'cov-valid-{rank}.pt')
        self.cov_valid = torch.load(path)
        self.cov = None

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def set_training(self):
        self.cov = self.cov_train

    def set_validation(self):
        self.cov = self.cov_valid
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        points_sz = sys.getsizeof(self.points.untyped_storage())
        self.set_training()
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        self.set_validation()
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

def process_epoch(model, dataloader, loss_fcn, penalty_fcn, optimizer):

    for points, cov in dataloader:

        # Forward pass
        preds = model(points[:,0], points[:,1])
        loss = loss_fcn(preds, cov)
        penalty = penalty_fcn(model.module.loads)
        objective = loss + penalty

        # Backward pass
        optimizer.zero_grad()
        objective.backward()
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


# def compute_objective(model, dataloader, alpha, penalty, rank, world_size):

#     # Compute and communicate loss
#     dataset = dataloader.dataset
#     preds = model(dataset.points[:,0], dataset.points[:,1])
#     objective = mse_loss(preds, dataset.cov)
#     if alpha > 0:
#         objective += alpha * penalty(model.module.loads)
#     if rank > 0: 
#         dist.send(objective, 0)
#     else: 
#         for r in range(1, world_size):
#             worker_objective = torch.zeros(1, dtype=torch.float64)
#             dist.recv(worker_objective, r)
#             objective += worker_objective.item()
#         return objective
    
    
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


def compute_loss_penalty(model, dataloader, loss_fcn, penalty_fcn, rank, world_size):

    def compute_loss_penalty_(dataloader):
        dataset = dataloader.dataset
        preds = model(dataset.points[:,0], dataset.points[:,1])
        loss = loss_fcn(preds, dataset.cov)
        penalty = penalty_fcn(model.module.loads)
        if rank > 0: 
            dist.send(loss, 0)
            dist.send(penalty, 0)
            return None, None
        else: 
            for r in range(1, world_size):
                worker_loss = torch.zeros(1, dtype=torch.float64)
                worker_penalty = torch.zeros(1, dtype=torch.float64)
                dist.recv(worker_loss, r)
                dist.recv(worker_penalty, r)
                loss += worker_loss.item()
                penalty += worker_penalty.item()
            return loss, penalty

    dataloader.set_training()
    train_loss, train_penalty = compute_loss_penalty_(dataloader)
    dataloader.set_validation()
    valid_loss, valid_penalty = compute_loss_penalty_(dataloader)

    return train_loss, train_penalty, valid_loss, valid_penalty


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
        batch_size, lr, patience, max_epochs, seed
    )


def train(
        rank: int, 
        world_size: int, 
        config: str,
        dir_out: str,
        grid_shape: List[int], 
        num_facs: int, 
        alpha: float,
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

    dataset = Dataset_(dir_cov, rank, world_size)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

    num_vars = multiply_list(grid_shape)
    path_init = os.path.join(dir_out, 'init_loads.pt')
    model = LowRankCovariance(num_vars, num_facs, path_init)
    model = DDP(model)

    diff_mat = create_second_difference_matrix(grid_shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fcn_ = partial(loss_fcn, num_vars=num_vars)
    penalty_fcn_ = partial(penalty_fcn, alpha=alpha, diff_mat=diff_mat)
    # objective_ = partial(objective, alpha=alpha, diff_mat=diff_mat)

    path_model = os.path.join(dir_out, 'model.pth')
    best_valid_loss = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    prev_train_loss = float('inf')
    lr = torch.tensor(lr)
    for epoch in range(max_epochs):
        
        dataloader.set_training()
        process_epoch_(model, dataloader, loss_fcn_, penalty_fcn_, optimizer)
        train_loss, train_penalty, valid_loss, _ = compute_loss_penalty(
            model, dataloader, 
            loss_fcn_, penalty_fcn_, 
            rank, world_size
        )
        
        # Rank-0 worker determines whether to stop and how to update lr
        if rank == 0:
            train_obj = train_loss + train_penalty
            print(f"epoch = {epoch + 1} | train_obj = {train_obj} | valid_loss = {valid_loss}")

            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                state_dict = model.state_dict()
                state_dict['loads'] = state_dict.pop('module.loads')  # replace DDP key
                torch.save({
                    'state_dict': state_dict,
                    'train_obj': train_obj,
                    'valid_loss': valid_loss
                }, path_model)
                epochs_waited = 0
            else:
                epochs_waited += 1
                if epochs_waited >= patience: 
                    print(f"Early stopping after {epoch + 1} epochs.")
                    early_stop = torch.tensor(True)

            # Update learning rate via bold driver
            # TODO: How to handle lr updates? Fixed?
            # lr *= 1.05 if train_loss < prev_train_loss else 0.5

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
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--grid_shape', type=int, nargs='+')
    parser.add_argument('--num_facs', type=int)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--delta', type=float)
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized']
    )
    parser.add_argument('--init_prop', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    # parser.add_argument('--opt', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set directories and paths
    dir_out = os.path.join('out', args.dir_out)
    dir_data = os.path.join(config.scratch_root, dir_out, 'data')
    dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
    dir_bench = os.path.join(dir_out, 'bench')
    path_init = os.path.join(dir_out, 'init_loads.pt')
    path_model = os.path.join(dir_out, 'model.pth')
    other_bench_path = os.path.join(dir_bench, 'other.csv')
    
    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Delete files from various directories
    remove_file(path_init)
    remove_file(path_model)

    # Multiprocessing configurations
    mp.set_start_method('spawn')


    # ---------- INITIALIZATION ---------- #
    print("Initializing loadings...")

    start = time.time()
    pca_svd_solvers = {
        'pca_full': 'full',
        'pca_arpack': 'arpack',
        'pca_randomized': 'randomized'
    }
    if args.init_method == 'random':
        num_vars = multiply_list(args.grid_shape)
        loads = torch.randn(args.num_facs, num_vars, generator=gen, dtype=torch.float64)
    else:
        dataloader = read_tensors(dir_data, 'data-train')
        initializer = PCALoadingInitializer(
            pca_svd_solvers[args.init_method], 
            args.num_facs, 
            args.init_prop, 
            gen
        )
        loads = initializer(dataloader)
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
                args.grid_shape, args.num_facs, args.alpha,
                args.batch_size, args.lr, 40, args.max_epochs, 
                seeds[rank], train, config.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

