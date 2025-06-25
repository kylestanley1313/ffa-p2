import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple

from config import load_config
from utils.data import (
    BasicDataLoader,
    DistributedCovarianceDataset,
    DistributedDatasetSampler
)
from utils.model import LowRankCovariance
from utils import (
    CODE_DIVERGENCE,
    gen_seeds, 
    init_process,
    load_yaml,
    loss_fcn,
    remove_file,
    write_rows_to_csv,
)



def process_epoch(
        model, 
        dataloader, 
        loss_fcn, 
        optimizer
    ):

    for points, cov in dataloader:

        # Forward pass
        preds = model(points)
        loss = loss_fcn(preds, cov)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def compute_objective(
        model, 
        dataloader, 
        loss_fcn, 
        rank, 
        world_size
    ) -> Tuple[torch.Tensor]:

    dataset = dataloader.dataset
    preds = model(dataset.points)
    loss = loss_fcn(preds, dataset.cov)
    if rank > 0: 
        dist.send(loss, 0)
        return None  # only rank-0 returns aggregated loss
    else: 
        for r in range(1, world_size):
            worker_loss = torch.zeros(1, dtype=torch.float32)
            dist.recv(worker_loss, r)
            loss += worker_loss.item()
        return loss


def train(
        rank: int, 
        world_size: int,
        dir_out: str,
        n_vars: int,
        n_comps: int, 
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
    
    # Set paths
    dir_cov = os.path.join(dir_out, 'cov')
    dir_idx = os.path.join(dir_out, f'idx-ddp-{world_size}')
    path_init = os.path.join(dir_out, f'init-loads-ddp-{world_size}.pt')
    path_model = os.path.join(dir_out, f'model-ddp-{world_size}.pth')

    dataset = DistributedCovarianceDataset(dir_cov, dir_idx, 'full', rank, world_size)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = LowRankCovariance(n_vars, n_comps, path_init)
    model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    last_objective = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    diverged = torch.tensor(False)
    if benchmark and rank == 0:
        bench_rows = []
    objective = compute_objective(  # initial objective
        model, dataloader, 
        loss_fcn,
        rank, world_size
    )
    if rank == 0:
        print(f"epoch = 0 | objective = {objective.item()}")
        bench_rows.append({
            'epoch': 0,
            'objective': objective.item(),
            'time': time.time()
        })
    for epoch in range(max_epochs):

        process_epoch(
            model, dataloader, 
            loss_fcn,
            optimizer
        )
        objective = compute_objective(
            model, dataloader, 
            loss_fcn,
            rank, world_size
        )
        
        # Rank-0 worker determines whether to stop and how to update lr
        if rank == 0:
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

        # Communicate early_stop and lr to non-zero ranks
        dist.barrier()
        dist.broadcast(diverged, 0)
        dist.broadcast(early_stop, 0)

        if benchmark:
            dist.barrier()
            if rank == 0:
                bench_rows.append({
                    'epoch': epoch + 1,
                    'objective': objective.item(),
                    'time': time.time()
                })

        if diverged or early_stop:
            break

    # Emit exit code and/or save model
    if rank == 0:

        if benchmark:
            path_bench = os.path.join(dir_out, f'epochs-ddp-{world_size}.csv')
            write_rows_to_csv(path_bench, bench_rows)

        if diverged: 
            print(f"Error: Divergence after {epoch + 1} epochs.")
            sys.exit(CODE_DIVERGENCE)

        else: 

            # Save model (even if no convergence)
            state_dict = model.state_dict()
            state_dict['loads'] = state_dict.pop('module.loads')  # replace DDP key
            torch.save(state_dict, path_model)
    
            if not early_stop:
                print(f"Warning: No convergence after {epoch + 1} epochs.")
            
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--tol', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Load design
    path_design = os.path.join(config.group_root, 'designs-lrcmc', f'{args.design}.yml')
    design = load_yaml(path_design)

    # Set directories and paths
    dir_out = os.path.join(config.group_root, 'out-lrcmc', args.design)
    dir_cov = os.path.join(dir_out, 'cov')
    dir_idx = os.path.join(dir_out, f'idx-ddp-{args.world_size}')
    path_init = os.path.join(dir_out, f'init-loads-ddp-{args.world_size}.pt')
    path_shared = os.path.join(config.dir_shared, f'{args.design}-{args.world_size}')
    path_model = os.path.join(dir_out, f'model-ddp-{args.world_size}.pth')
    path_bench = os.path.join(dir_out, f'epochs-ddp-{args.world_size}.csv')

    # File cleanup
    remove_file(path_init)
    remove_file(path_shared)
    remove_file(path_model)
    remove_file(path_bench)

    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)


    # ---------- INITIALIZE ---------- #

    # Initialize loadings by sampling repeatedly from U(-0.5, 0.5)
    init_loads = torch.rand(design['n_vars'], design['rank'], generator=gen) - 0.5
    torch.save(init_loads, path_init)


    # ---------- ESTIMATION ---------- #
    print("Fitting model...")

    mp.set_start_method('spawn')
    processes = []
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(rank, args.world_size, train, path_shared, config.backend),
            kwargs={
                'dir_out': dir_out,
                'n_vars': design['n_vars'],
                'n_comps': design['rank'],
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

