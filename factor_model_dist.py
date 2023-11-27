import argparse
import inspect
import math
import numpy as np
import os
import random
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Generator, List

# PLAN: 
#   - Implement for 1-dim FFA
#   - Implement for D-dim FFA

# TODO: 
#   - Utility functions: tensor_i32, tensor_f64, tensor_b
#   - Figure out what's going on with `gloo::EnforceNotMet` error
#       * Can't seem to handle them with try-except like standard errors
#       * To reproduce, try sending a tensor of one type to a tensor initialized
#         as other type. 

# NOTE: Tips for band-aiding multi-processing issues: 
#   - restart shell
#   - kill processes by iteratively calling ctrl-c (but some still linger -- set timeout? (see zsh terminal))



# -------------------- UTILITIES -------------------- #

class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


def gen_seeds(gen, size):
    seeds = torch.randint(
        high=torch.iinfo(torch.int32).max, 
        size=(size,), 
        generator=gen, 
        dtype=torch.int32
    )
    if size == 1: 
        return seeds.tolist()[0]
    else: 
        return seeds.tolist()


# -------------------- DATA -------------------- #


def simulate_ffm_data(
        loadings: torch.Tensor,
        err_sds: torch.Tensor, 
        num_train: int,
        num_val: int,
        batch_size: int,
        gen: torch.Generator = torch.Generator(),
    ):
    num_vars, num_facs = loadings.shape
    n = num_train + num_val
    samps = 0
    while samps < n: 
        n_batch = min(batch_size, n - batch_size)
        facs = torch.normal(0, 1, (num_facs, n_batch), generator=gen)
        errs = torch.normal(0, 1, (num_vars, n_batch), generator=gen)
        errs *= err_sds
        data = torch.matmul(loadings, facs) + errs
        yield data
        samps += batch_size


def write_data(data: Generator, dir: str):
    for i, batch in enumerate(data):
        path = os.path.join(dir, f'data-{i}.pt')
        torch.save(batch, path)


def read_data(dir):
    for name in os.listdir(dir):
        yield torch.load(os.path.join(dir, name))

        



# -------------------- DISTRIBUTION -------------------- #

# def init_process(addr, port, rank, world_size, dir_data, delta, fcn, backend):
#     os.environ['MASTER_ADDR'] = addr  # TODO: How to choose ADDR/PORT?
#     os.environ['MASTER_PORT'] = port
#     dist.init_process_group(backend, rank=rank, world_size=world_size)
#     fcn(rank, world_size, dir_data, delta)

def init_process(rank, world_size, dir_data, delta, num_facs, max_epochs, seeds, fcn, backend):
    path = '/tmp/sharedfile'
    if os.path.exists(path):
        os.remove(path)
    dist.init_process_group(
        backend, init_method=f'file://{path}',
        rank=rank, world_size=world_size
    )
    fcn(rank, world_size, dir_data, delta, num_facs, max_epochs, seeds)


# -------------------- RUN -------------------- #

def fair_allocate(num_items: int, num_groups: int) -> List[int]:
    """Evenly distributes num_items across num_groups."""
    out = [num_items // num_groups] * num_groups
    remainder = num_items % num_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out
 

def gen_train_points(grid_size: int, delta: float) -> torch.Tensor:
    """Generates covariance matrix training points for data of a given 
    grid_size (M) and delta (bandwidth)."""
    grid = torch.arange(grid_size, dtype=torch.int32)
    points = torch.cartesian_prod(grid, grid)
    bandwidth = math.ceil(grid_size * delta)
    keep = points[:,1] < points[:,0] - bandwidth
    return points[keep,:]


def block_train_points(
        train_points: torch.Tensor, 
        grid_size: int, 
        num_workers: int,
        gen: torch.Generator
    ) -> torch.Tensor:
    """Assigns (lower-triangular) blocks to training points."""
    block_cnts = fair_allocate(grid_size, 2*num_workers)
    idx = 0
    blocks = torch.zeros(grid_size, dtype=torch.int32)
    for b, cnt in enumerate(block_cnts):
        blocks[idx:(idx+cnt)] = torch.ones(cnt) * b
        idx += cnt
    blocks = blocks[torch.randperm(grid_size, generator=gen)]
    blocks = blocks[train_points]

    # Make blocks[k,0] > blocks[k,1] for all k to make blocks lower triangular
    temp = blocks.clone()
    mask = blocks[:,0] < blocks[:,1]
    blocks[mask, 0] = temp[mask, 1]
    blocks[mask, 1] = temp[mask, 0]
    return torch.cat((train_points, blocks), 1)  # cols: x[0], x[1], b[0], b[1]


def gen_strata(num_workers):

    if num_workers == 2: 
        strata = [
            ((0, 0), (1, 1)),
            ((2, 2), (3, 3)),
            ((1, 0), (3, 2)),
            ((2, 0), (3, 1)),
            ((3, 0), (2, 1)),
        ]
    else: 
        raise Exception(f"Function does not support num_workers = {num_workers}")

    return strata


def run(rank, world_size, dir_data, delta, num_facs, max_epochs, seeds):

    seed = seeds[rank]
    gen = torch.Generator().manual_seed(seed)
    path = os.path.join(dir_data, 'data-0.pt')
    grid_size = torch.load(path).shape[0]
    num_workers = world_size - 1
    done = torch.zeros(1, dtype=torch.bool)

    if rank == 0:

        # Generate and block training points, then send to workers
        points = gen_train_points(grid_size, delta)
        points = block_train_points(points, grid_size, num_workers, gen)  # NOTE: points may be prohibitively large for one machine (stream?)
        print(f"Rank {rank} points = \n{points}\n")
        strata = gen_strata(num_workers)
        for r in range(1, world_size):
            blocks = [s[r - 1] for s in strata]
            mask = torch.zeros(len(points), dtype=torch.bool)
            for b in blocks: 
                temp = points[:,2:4] == torch.tensor(b)
                temp = torch.logical_and(temp[:,0], temp[:,1])
                mask = torch.logical_or(mask, temp)
            worker_points = points[mask]
            dist.send(torch.tensor(worker_points.shape, dtype=torch.int32), r)
            dist.send(worker_points, r)

        # Initialize and send L
        # TODO: Better initialization? SVD on data matrix?
        l = torch.randn(
            grid_size, num_facs, 
            generator=gen, 
            dtype=torch.float64, 
            requires_grad=True
        )
        for r in range(1, world_size):
            dist.send(l, r)


        epochs = 0
        rand = random.Random(seed)
        while True:  # > epoch

            # Send blocks to workers
            rand.shuffle(strata)
            for stratum in strata:  # > subepoch

                # Send blocks to workers
                for r in range(1, num_workers + 1):
                    block = stratum[r - 1]
                    dist.send(torch.tensor(block, dtype=torch.int32), r)

                # Receive and combine updated L blocks from each worker
                # NOTE: Must wait until all updates have been received before proceeding

                # Send updated L to each worker

            # Send updated convergence status to workers
            # TODO: Parallelized convergence evaluation
            epochs += 1
            if epochs == max_epochs:
                done[0] = True
            for r in range(1, num_workers + 1):
                dist.send(tensor=done, dst=r)
            if done[0]:
                break


        # Optimize (epoch): [while not converged]
        #   - Choose stratum sequence
        #   - For stratum in stratum sequence: 
        #       * Send blocks
        #       * Receive updated L blocks
        #       * Stitch together L blocks
        #       * Send updated L
        #   - Send end of epoch message
        #   - Check convergence status
        #   - Send convergence status
        #   - If converged; exit(?) process
    
    else: 

        # Receive training points from main process
        shape = torch.zeros(2, dtype=torch.int32)
        dist.recv(shape, 0)
        shape = tuple(shape.tolist())
        points = torch.zeros(shape, dtype=torch.int32)
        dist.recv(tensor=points, src=0)

        # Receive loadings from main process
        l = torch.zeros(
            grid_size, num_facs, 
            dtype=torch.float64, 
            requires_grad=True
        )
        dist.recv(tensor=l, src=0)

        # Compute covariance of training points
        # NOTE: c_{ij} = \frac{1}{N-1} [T_1 - \frac{T_2 T_3}{N}]
        #   where T_1 = sum(x_i * x_j)
        #         T_2 = sum(x_i)
        #         T_3 = sum(x_j)
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data = read_data(dir_data)
        for batch in data: 
            n += batch.shape[1]
            for i in range(num_points): 
                row, col = points[i,0:2]
                t1[i] += torch.sum(batch[row,:] * batch[col,:])
                t2[i] += torch.sum(batch[row,:])
                t3[i] += torch.sum(batch[col,:])
        cov = (t1 - t2 * t3 / n) / (n - 1)
        print(f"Rank {rank} covariance: {cov}\n")

        # Optimize until main process says to stop
        strata_per_epoch = 2 * num_workers + 1
        block = torch.zeros(2, dtype=torch.int32)
        while True: 

            # Iteratively perform SGD on blocks then send updated L to main process
            for _ in range(strata_per_epoch):
                dist.recv(block, 0)
                print(f"----- Rank {rank} | block = {block.tolist()} -----\n")

                # Perform SGD on block (pure? mini-batch?)
                mask = torch.logical_and(
                    points[:,2] == block[0], 
                    points[:,3] == block[1]
                )
                points_block = points[mask, 0:2]
                cov_block = cov[mask]
                l_b1 = l[points_block[:,0],:]
                l_b2 = l[points_block[:,1],:]
                print(f"Rank {rank} points_block = \n{points_block}\n")
                print(f"Rank {rank} l_b1 = \n{l_b1}\n")
                print(f"Rank {rank} l_b2 = \n{l_b2}\n")

                # Send L to main process

            # Receive updated L

            # Compute local loss 

            dist.recv(done, 0)
            print(f"done = {done[0]}")
            if done[0]:
                print("DONE!")
                break


        # Optimize:
        #   - Receive L
        #   - If end of epoch:
        #       * compute local loss on all of this worker's blocks
        #       * send local loss
        #   - Receive convergence status 
        #   - If converged: exit(?) process
        #   - Receive block
        #   - Run SGD on block
        #   - Send updated block to main

    dist.destroy_process_group()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='gloo')
    parser.add_argument('-ws', '--world_size', type=int)
    parser.add_argument('-sn', '--simulate_new', action = 'store_true')
    args = parser.parse_args()

    # Configure globals
    # addr = '127.0.0.1'
    # port = '29500'
    dir_data = './data/ffa'
    delta = 0.1
    num_facs = 2
    max_epochs = 2
    seed = 12345
    gen = torch.Generator().manual_seed(seed)
    seeds = gen_seeds(gen, args.world_size)


    # ---------- SIMULATION ---------- #

    if args.simulate_new: 

        print(f"Simulating new data...")

        sim_seed = gen_seeds(gen, 1)
        gen = torch.Generator().manual_seed(sim_seed)

        loadings = torch.tensor([
            [3, 0.1],
            [-2, 0.3],
            [0.1, 2.5], 
            [-0.2, 4],
            [-0.3, -3],
            [2, 0.5],
            [-2, 0.3],
            [0.1, 2.5], 
            [-0.2, 4],
            [-0.3, -3],
            [2, 0.5]
        ])
        err_sds = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3])
        data = simulate_ffm_data(
            loadings, 
            err_sds,
            num_train=20,
            num_val=10,
            batch_size=5,
            gen=gen
        )
        write_data(data, dir_data)

        print("DONE!")
    else: 
        print("Using existing simulated data.")


    # ---------- DISTRIBUTED RUN ---------- #

    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, 
                dir_data, delta, num_facs, max_epochs, seeds,
                run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()