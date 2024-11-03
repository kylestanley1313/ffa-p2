import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List

from config import load_config
from utils.data import (
    DistributedStratifiedCovarianceDataset,
    DistributedStratifiedDatasetBatchSampler,
    StratifiedDataLoader,
)
from utils import (
    CODE_DIVERGENCE,
    gen_seeds, 
    init_process,
    loss_fcn,
    remove_file,
    write_rows_to_csv,
)
from utils.model import LowRankCovariance
    

def sync_model(
        rank: int, 
        world_size: int, 
        model: LowRankCovariance, 
        points: torch.Tensor
    ) -> None:

    idx0 = torch.unique(points[:,0])
    idx1 = torch.unique(points[:,1])
    idx = torch.unique(torch.cat((idx0, idx1)))
    sz = torch.tensor([len(idx)], dtype=torch.int32)
    loads = model.get_loads(idx)
    other_ranks = [r for r in range(world_size) if r != rank]
    
    # Send/receive `sz` to/from all other ranks
    sz_in = {r: torch.zeros(1, dtype=torch.int32) for r in other_ranks}
    reqs_sz_out = {}
    reqs_sz_in = {}
    for r in other_ranks:
        reqs_sz_out[r] = dist.isend(sz, r)
        reqs_sz_in[r] = dist.irecv(sz_in[r], r)
    for r in other_ranks:
        reqs_sz_out[r].wait()
        reqs_sz_in[r].wait() 

    # Send/receive `idx` and `loads` to/from all other ranks
    idx_in = {
        r: torch.zeros(sz_in[r].item(), dtype=torch.int32) 
        for r in other_ranks
    }
    reqs_idx_out = {}
    reqs_idx_in = {}
    loads_in = {
        r: torch.zeros(sz_in[r].item(), model.n_facs, dtype=torch.float64) 
        for r in other_ranks
    }
    reqs_loads_out = {}
    reqs_loads_in = {}
    for r in other_ranks:
        reqs_idx_out[r] = dist.isend(idx, r)
        reqs_idx_in[r] = dist.irecv(idx_in[r], r)
        reqs_loads_out[r] = dist.isend(loads, r)
        reqs_loads_in[r] = dist.irecv(loads_in[r], r)
    for r in other_ranks:
        reqs_idx_out[r].wait()
        reqs_idx_in[r].wait()
        reqs_loads_out[r].wait()
        reqs_loads_in[r].wait()

    # Update model
    for r in other_ranks:
        model.set_loads(loads_in[r], idx_in[r])


def broadcast_model(model: LowRankCovariance, rank: int, src: int) -> None:
    if rank == src:
        loads = model.get_loads()
    else:
        loads = torch.zeros_like(model.get_loads())
    dist.broadcast(loads, src)
    model.set_loads(loads)


def process_epoch(
        model, 
        dataloader, 
        objective, 
        optimizer, 
        gen,
        n_strata, 
        rank, 
        world_size
    ):

    # Broadcast stratum sequence from rank 0
    if rank == 0:
        strat_seq = torch.randperm(n_strata, generator=gen, dtype=torch.int32)
    else:
        strat_seq = torch.zeros(n_strata, dtype=torch.int32)
    dist.broadcast(strat_seq, 0)

    for s in strat_seq:  # > subepoch
        dataloader.set_stratum(s.item())

        for points, cov in dataloader:

            # Forward pass
            preds = model(points)
            loss = objective(preds, cov)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync model
        dist.barrier()
        sync_model(rank, world_size, model, points)


def compute_objective(model, dataloader, loss_fcn, rank, world_size):

    dataloader.set_full_dataset()
    preds = model(dataloader.dataset.points)
    loss = loss_fcn(preds, dataloader.dataset.cov)
    if rank > 0: 
        dist.send(loss, 0)
        return None
    else: 
        for r in range(1, world_size):
            worker_loss = torch.zeros(1, dtype=torch.float64)
            dist.recv(worker_loss, r)
            loss += worker_loss.item()
        return loss


def train(
        rank: int, 
        world_size: int, 
        dir_out: str,
        dir_out_scratch: str,
        split: str,
        n_vars: int, 
        n_facs: int, 
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
    n_strata = 2 * world_size + 1

    # Set directories
    dir_cov = os.path.join(dir_out_scratch, 'cov')
    dir_idx = os.path.join(dir_out_scratch, 'idx-dssgd')
    path_model = os.path.join(dir_out, f'model-dssgd-{split}.pth')

    dataset = DistributedStratifiedCovarianceDataset(dir_cov, dir_idx, split, rank, world_size)
    batch_sampler = DistributedStratifiedDatasetBatchSampler(dataset, batch_size, gen)
    dataloader = StratifiedDataLoader(dataset, batch_sampler=batch_sampler)

    path_init = os.path.join(dir_out, f'init-loads-{split}.pt')
    model = LowRankCovariance(n_vars, n_facs, path_init)
    broadcast_model(model, rank, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    last_objective = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    diverged = torch.tensor(False)
    # prev_train_loss = float('inf')
    # lr = torch.tensor(lr)
    if benchmark:
        bench_rows = []
    for epoch in range(max_epochs):

        if benchmark:
            dist.barrier()
            start = time.time()

        process_epoch(
            model, dataloader, loss_fcn, optimizer, 
            gen, n_strata, rank, world_size
        )
        objective = compute_objective(
            model, dataloader, loss_fcn, 
            rank, world_size
        )
        
        # Rank-0 worker determines whether to stop and how to update lr
        if rank == 0:
            print(f"epoch = {epoch + 1} | objective = {objective}")

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
            # lr *= 1.05 if train_loss < prev_train_loss else 0.5
        
        # Communicate early_stop and lr to non-zero ranks
        dist.barrier()
        dist.broadcast(diverged, 0)
        dist.broadcast(early_stop, 0)

        if benchmark: 

            # Compute epoch time
            dist.barrier()
            epoch_time = time.time() - start

            # Aggregate times then choose maximum
            if rank > 0: 
                dist.send(torch.tensor(epoch_time, dtype=torch.float64), 0)
            else: 
                max_epoch_time = epoch_time
                for r in range(1, world_size):
                    worker_epoch_time = torch.zeros(1, dtype=torch.float64)
                    dist.recv(worker_epoch_time, r)
                    max_epoch_time = max(max_epoch_time, worker_epoch_time.item())
                bench_rows.append({
                    'epoch': epoch + 1,
                    'objective': objective.item(),
                    'time': max_epoch_time
                })  

        if diverged or early_stop:
            break
        # dist.broadcast(lr, 0)
        # optimizer.param_groups[0]['lr'] = lr.item()

    # Emit exit code and/or save model
    if rank == 0:

        if benchmark:
            path_bench = os.path.join(dir_out, 'bench', 'epochs-dssgd.csv')
            write_rows_to_csv(path_bench, bench_rows)

        if diverged: 
            print(f"Error: Divergence after {epoch + 1} epochs.")
            sys.exit(CODE_DIVERGENCE)

        else: 

            # Save model (even if no convergence)
            torch.save(model.state_dict(), path_model)

            if not early_stop:
                print(f"Warning: No convergence after {epoch + 1} epochs.")
            

    dist.destroy_process_group()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--split', type=str, choices=['full', 'train', 'valid'])
    parser.add_argument('--n_vars', type=int)
    parser.add_argument('--n_facs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--tol', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Configure globals
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Set directories and paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_cov = os.path.join(args.dir_out_scratch, 'cov-dssgd')
    dir_bench = os.path.join(args.dir_out, 'bench')
    path_init = os.path.join(args.dir_out, f'init-loads-{args.split}.pt')
    path_model = os.path.join(args.dir_out, f'model-dssgd-{args.split}.pth')
    other_bench_path = os.path.join(dir_bench, 'other-dssgd.csv')
    suffix = args.dir_out.split('out/')[-1].replace('/', '_')
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')

    # Delete old model
    remove_file(path_model)

    # Multiprocessing configurations
    mp.set_start_method('spawn')


    # ---------- ESTIMATION ---------- #
    print("Fitting model...")

    remove_file(path_shared)
    processes = []
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(rank, args.world_size, train, path_shared, config.backend),
            kwargs={
                'dir_out': args.dir_out,
                'dir_out_scratch': args.dir_out_scratch,
                'split': args.split,
                'n_vars': args.n_vars,
                'n_facs': args.n_facs,
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

