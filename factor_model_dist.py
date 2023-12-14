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
from torch.nn.functional import mse_loss
from typing import Generator, List

from utils import read_tensors



# PLAN: 
#   - Implement for 1-dim FFA
#   - Implement for D-dim FFA
#   - Can we wedge factor model into standard PyTorch framework? This first
#     iteration makes use only of torch.distributed machinery. 


# NOTE: If you must ctrl-c to terminate execution, you still need to kill
# processes on the command line: 
#   $ ps -ef | grep ffa-p2-priv
#   $ kill -9 <pid>
# TODO: Find more permanent solution to this.



# -------------------- UTILITIES -------------------- #


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



# -------------------- DISTRIBUTION -------------------- #

def init_process(rank, world_size, dir_data, delta, num_facs, lr, max_epochs, seeds, fcn, backend):
    path = '/tmp/sharedfile'
    if os.path.exists(path):
        os.remove(path)
    dist.init_process_group(
        backend, init_method=f'file://{path}',
        rank=rank, world_size=world_size
    )
    fcn(rank, world_size, dir_data, delta, num_facs, lr, max_epochs, seeds)


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
        points: torch.Tensor, 
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
    b0 = blocks[points[:,0]]
    b1 = blocks[points[:,1]]
    blocks = torch.column_stack((b0, b1))

    # Make blocks[k,0] > blocks[k,1] for all k to make blocks lower triangular
    temp_blocks = blocks.clone()
    temp_points = points.clone()
    mask = blocks[:,0] < blocks[:,1]
    blocks[mask, 0] = temp_blocks[mask, 1]
    blocks[mask, 1] = temp_blocks[mask, 0]
    points[mask, 0] = temp_points[mask, 1]
    points[mask, 1] = temp_points[mask, 0]
    return torch.cat((points, blocks), 1)  # cols: x[0], x[1], b[0], b[1]


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



def run(rank, world_size, dir_data, delta, num_facs, lr, max_epochs, seeds):

    seed = seeds[rank]
    gen = torch.Generator().manual_seed(seed)
    path = os.path.join(dir_data, 'data-0.pt')
    grid_size = torch.load(path).shape[0]
    num_workers = world_size - 1

    if rank == 0:

        # Generate and block training points, then send to workers
        print(f"Rank {rank}: Allocating training points...")
        points = gen_train_points(grid_size, delta)
        points = block_train_points(points, grid_size, num_workers, gen)  # NOTE: points may be prohibitively large for one machine (stream?)
        num_points = len(points)
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

        # Initialize L via SVD(?)
        # TODO: Better initialization? SVD on data matrix?
        # NOTE: Only the workers needs gradients. Should only have to pass data between processes.
        print(f"Rank {rank}: Initializing loadings...")
        l = torch.randn(
            grid_size, num_facs, 
            generator=gen, 
            dtype=torch.float64
        )

    else: 

        # Receive training points from main process
        print(f"Rank {rank}: Receiving training points...")
        shape = torch.zeros(2, dtype=torch.int32)
        dist.recv(shape, 0)
        points = torch.zeros(shape.tolist(), dtype=torch.int32)
        dist.recv(tensor=points, src=0)

        # Compute covariance of training points
        # NOTE: c_{ij} = \frac{1}{N-1} [T_1 - \frac{T_2 T_3}{N}]
        #   where T_1 = sum(x_i * x_j)
        #         T_2 = sum(x_i)
        #         T_3 = sum(x_j)
        print(f"Rank {rank}: Computing covariances...")
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data = read_tensors(dir_data, 'data')
        for batch in data: 
            n += batch.shape[1]
            for i in range(num_points): 
                row, col = points[i,0:2]
                t1[i] += torch.sum(batch[row,:] * batch[col,:])
                t2[i] += torch.sum(batch[row,:])
                t3[i] += torch.sum(batch[col,:])
        cov = (t1 - t2 * t3 / n) / (n - 1)

        # Initialize L as empty matrix
        l = torch.zeros(
            grid_size, num_facs, 
            dtype=torch.float64
        )

    # Broadcast L
    dist.broadcast(tensor=l, src=0)

    # Optimize
    print(f"Rank {rank}: Optimizing...")
    epoch = 0
    done = False
    while not done:  # > epoch

        # Set stratum sequence and broadcast to blocks
        if rank == 0:
            rand = random.Random(seed)
            rand.shuffle(strata)
        else:
            strata_per_epoch = 2 * num_workers + 1
            strata = [None] * strata_per_epoch
        dist.broadcast_object_list(strata, src=0)

        for stratum in strata:  # > subepoch

            if rank != 0:

                # Identify points and covariances associated with block
                block = stratum[rank - 1]
                mask = torch.logical_and(
                        points[:,2] == block[0], 
                        points[:,3] == block[1]
                )
                points_block = points[mask, 0:2]
                cov_block = cov[mask]

                # Perform SGD on block
                l.requires_grad_(True)
                optimizer = torch.optim.SGD([l], lr)
                seq = torch.randperm(len(points_block), generator=gen)
                for s in seq: 

                    # Compute local loss
                    point = points_block[s,:]
                    pred = torch.matmul(l[point[0],:], l[point[1],:])
                    loss = mse_loss(pred, cov_block[s])

                    # Backward pass and step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if block[0] == block[1]:
                    
                    # Get updated L block
                    idx0 = torch.unique(points_block[:,0])
                    idx1 = torch.unique(points_block[:,1])
                    idx = torch.unique(torch.cat((idx0, idx1)))
                    lb = l[idx,:].data

                    # Send L block to main process
                    sz = torch.tensor([len(idx)], dtype=torch.int32)
                    req_sz = dist.isend(sz, 0)
                    req_idx = dist.isend(idx, 0)
                    req_lb = dist.isend(lb, 0)
                    req_sz.wait()
                    req_idx.wait()
                    req_lb.wait()
                    
                else:

                    # Get updated L blocks
                    idx0 = torch.unique(points_block[:,0])
                    idx1 = torch.unique(points_block[:,1])
                    lb0 = l[idx0,:].data
                    lb1 = l[idx1,:].data

                    # Send L blocks to main process
                    sz0 = torch.tensor([len(idx0)], dtype=torch.int32)
                    sz1 = torch.tensor([len(idx1)], dtype=torch.int32)
                    req_sz0 = dist.isend(sz0, 0)
                    req_sz1 = dist.isend(sz1, 0)
                    req_idx0 = dist.isend(idx0, 0)
                    req_idx1 = dist.isend(idx1, 0)
                    req_lb0 = dist.isend(lb0, 0)
                    req_lb1 = dist.isend(lb1, 0)
                    req_sz0.wait()
                    req_sz1.wait()
                    req_idx0.wait()
                    req_idx1.wait()
                    req_lb0.wait()
                    req_lb1.wait()

            else: 

                # Receive updated L blocks from workers
                num_l_blocks = len(set(b for block in stratum for b in block))
                idx = [None] * num_l_blocks
                lb = [None] * num_l_blocks
                reqs_idx = [None] * num_l_blocks
                reqs_lb = [None] * num_l_blocks
                for r in range(1, world_size):
                    block = stratum[r - 1]

                    if block[0] == block[1]:
                        i = r - 1

                        sz = torch.zeros(1, dtype=torch.int32)
                        req_sz = dist.irecv(sz, src=r)
                        req_sz.wait()

                        idx[i] = torch.zeros(sz.item(), dtype=torch.int32)
                        reqs_idx[i] = dist.irecv(idx[i], src=r)                        

                        lb[i] = torch.zeros(sz.item(), num_facs, dtype=torch.float64)
                        reqs_lb[i] = dist.irecv(lb[i], src=r)     

                    else: 
                        i = 2*(r - 1)
                        
                        sz0 = torch.zeros(1, dtype=torch.int32)
                        sz1 = torch.zeros(1, dtype=torch.int32)
                        req_sz0 = dist.irecv(sz0, src=r)
                        req_sz1 = dist.irecv(sz1, src=r)
                        req_sz0.wait()
                        req_sz1.wait()
                        
                        idx[i] = torch.zeros(sz0.item(), dtype=torch.int32)
                        idx[i+1] = torch.zeros(sz1.item(), dtype=torch.int32)
                        reqs_idx[i] = dist.irecv(idx[i], src=r)
                        reqs_idx[i+1] = dist.irecv(idx[i+1], src=r)
                        
                        lb[i] = torch.zeros(sz0.item(), num_facs, dtype=torch.float64)
                        lb[i+1] = torch.zeros(sz1.item(), num_facs, dtype=torch.float64)
                        reqs_lb[i] = dist.irecv(lb[i], src=r) 
                        reqs_lb[i+1] = dist.irecv(lb[i+1], src=r) 
                
                # Wait for each request to complete
                for r_idx, r_lb in zip(reqs_idx, reqs_lb):
                    r_idx.wait()
                    r_lb.wait()

                # Update L with L blocks
                for idx_, lb_ in zip(idx, lb):
                    l[idx_, :] = lb_
                
            # Broadcast L to all workers
            dist.barrier()
            dist.broadcast(l, src=0)


        # Check Convergence
        if rank != 0:
            l0 = l[points[:,0], :].data
            l1 = l[points[:,1], :].data
            pred = torch.sum(l0 * l1, dim=1)
            loss = mse_loss(pred, cov)
            dist.send(loss, 0)
        else:
            losses = torch.zeros(num_workers, dtype=torch.float64)
            for r in range(1, world_size):
                loss = torch.zeros(1, dtype=torch.float64)
                dist.recv(loss, r)
                losses[r - 1] = loss.item()
            loss = torch.sum(losses) / num_points
            print(f"Epoch {epoch} loss = {loss}")


        # Determine whether to stop
        epoch += 1
        if epoch == max_epochs:
            done = True


    dist.destroy_process_group()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='gloo')
    parser.add_argument('-ws', '--world_size', type=int)
    args = parser.parse_args()

    # Configure globals
    # addr = '127.0.0.1'
    # port = '29500'
    dir_data = './data/ffa'
    delta = 0.1
    num_facs = 2
    lr = 0.01
    max_epochs = 20
    seed = 12345
    gen = torch.Generator().manual_seed(seed)
    seeds = gen_seeds(gen, args.world_size)


    # ---------- DISTRIBUTED RUN ---------- #

    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, 
                dir_data, delta, num_facs, lr, max_epochs, seeds,
                run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()