import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils.data import (
    BasicDataLoader,
    DistributedCovarianceDataset,
    DistributedDatasetSampler
)
from utils.model import LowRankCovariance
from utils import (
    CODE_DIVERGENCE,
    CODE_NO_CONVERGENCE,
    create_second_difference_matrix, 
    gen_seeds, 
    init_process,
    loss_fcn,
    multiply_list,
    penalty_fcn,
    remove_file,
)



def process_epoch(
        model, 
        dataloader, 
        loss_fcn, 
        penalty_fcn, 
        optimizer
    ):

    for points, cov in dataloader:

        # Forward pass
        preds = model(points)
        loss = loss_fcn(preds, cov)
        penalty = penalty_fcn(model.module.loads)
        objective = loss + penalty

        # Backward pass
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()



def compute_objective(
        model, 
        dataloader, 
        loss_fcn, 
        penalty_fcn, 
        rank, 
        world_size
    ) -> Tuple[torch.Tensor]:

    dataset = dataloader.dataset
    preds = model(dataset.points)
    loss = loss_fcn(preds, dataset.cov)
    penalty = penalty_fcn(model.module.loads)
    if rank > 0: 
        dist.send(loss, 0)
        dist.send(penalty, 0)
        return None, None  # only rank-0 returns aggregated loss/penalty
    else: 
        for r in range(1, world_size):
            worker_loss = torch.zeros(1, dtype=torch.float64)
            worker_penalty = torch.zeros(1, dtype=torch.float64)
            dist.recv(worker_loss, r)
            dist.recv(worker_penalty, r)
            loss += worker_loss.item()
            penalty += worker_penalty.item()
        return loss, penalty


def train(
        rank: int, 
        world_size: int, 
        dir_out: str,
        dir_out_scratch: str,
        split: str,
        sz_space: List[int], 
        n_facs: int, 
        alpha: float,
        batch_size: int,
        lr: float, 
        tol: float,
        patience: int,
        max_epochs: int, 
        benchmark: bool,
        seed: int
    ) -> None:

    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(seed)
    
    # Set directories
    dir_cov = os.path.join(dir_out_scratch, 'cov-dsgd')
    dir_bench = os.path.join(dir_out, 'bench')
    path_model = os.path.join(dir_out, f'model-dsgd-{split}.pth')

    # Get benchmarking wrappers
    process_epoch_ = time_dist_fcn(
        fcn=process_epoch,
        dir=dir_bench,
        prefix='process_epoch',
        benchmark=benchmark
    )
    Dataset_ = size_dist_obj(
        init=DistributedCovarianceDataset,
        dir=dir_bench,
        prefix='dataset',
        benchmark=benchmark
    )

    dataset = Dataset_(dir_cov, split, rank, world_size)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

    n_vars = multiply_list(sz_space)
    path_init = os.path.join(dir_out, f'init-loads-{split}.pt')
    model = LowRankCovariance(n_vars, n_facs, path_init)
    model = DDP(model)

    diff_mat = create_second_difference_matrix(sz_space)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fcn_ = partial(loss_fcn, n_vars=n_vars)
    penalty_fcn_ = partial(penalty_fcn, alpha=alpha, diff_mat=diff_mat)

    last_objective = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    diverged = torch.tensor(False)
    # prev_train_loss = float('inf')
    # lr = torch.tensor(lr)
    for epoch in range(max_epochs):
        process_epoch_(
            model, dataloader, 
            loss_fcn_, penalty_fcn_, 
            optimizer
        )
        loss, penalty = compute_objective(
            model, dataloader, 
            loss_fcn_, penalty_fcn_, 
            rank, world_size
        )
        
        # Rank-0 worker determines whether to stop and how to update lr
        if rank == 0:
            objective = loss + penalty
            print(f"epoch = {epoch + 1} | objective = {objective.item()}")

            # Handle divergence
            if torch.isnan(objective) or torch.isinf(objective):
                diverged = torch.tensor(True)

            else:
                # Handle early stopping
                if abs(objective - last_objective) > tol:
                    epochs_waited = 0
                else: 
                    epochs_waited += 1
                    if epochs_waited >= patience: 
                        print(f"Early stopping after {epoch + 1} epochs.")
                        early_stop = torch.tensor(True)
                last_objective = objective

            # Update learning rate via bold driver
            # TODO: How to handle lr updates? Fixed?
            # lr *= 1.05 if train_loss < prev_train_loss else 0.5

        # Communicate early_stop and lr to non-zero ranks
        dist.barrier()
        dist.broadcast(diverged, 0)
        dist.broadcast(early_stop, 0)
        if diverged or early_stop:
            break
        # dist.broadcast(lr, 0)
        # optimizer.param_groups[0]['lr'] = lr.item()

    # TODO: Consider using exit codes to handle divergence/no-convergence/etc. at script level
    if rank == 0:

        if diverged: 
            print(f"Error: Divergence after {epoch + 1} epochs.")
            sys.exit(CODE_DIVERGENCE)

        else: 
            if not early_stop:
                print(f"Warning: No convergence after {epoch + 1} epochs.")
                sys.exit(CODE_NO_CONVERGENCE)
            
            # Save model (even if no convergence)
            state_dict = model.state_dict()
            state_dict['loads'] = state_dict.pop('module.loads')  # replace DDP key
            torch.save(state_dict, path_model)

    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--split', type=str, choices=['full', 'train', 'valid'])
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument('--n_facs', type=int)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--tol', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set directories and paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_cov = os.path.join(args.dir_out_scratch, 'cov-dsgd')
    dir_bench = os.path.join(args.dir_out, 'bench')
    path_init = os.path.join(args.dir_out, f'init-loads-{args.split}.pt')
    path_model = os.path.join(args.dir_out, f'model-dsgd-{args.split}.pth')
    other_bench_path = os.path.join(dir_bench, 'other-dsgd.csv')
    
    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)


    # ---------- ESTIMATION ---------- #
    print("Fitting model...")

    suffix = args.dir_out.split('out/')[-1].replace('/', '_')
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')
    remove_file(path_shared)
    remove_file(path_model)

    mp.set_start_method('spawn')
    processes = []
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(rank, args.world_size, train, path_shared, config.backend),
            kwargs={
                'dir_out': args.dir_out,
                'dir_out_scratch': args.dir_out_scratch,
                'split': args.split,
                'sz_space': args.sz_space,
                'n_facs': args.n_facs,
                'alpha': args.alpha,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'tol': args.tol,
                'patience': args.patience,
                'max_epochs': args.max_epochs,
                'benchmark': args.benchmark,
                'seed': seeds[rank]
            }
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Exit script according to status of 0th worker
    sys.exit(processes[0].exitcode)

