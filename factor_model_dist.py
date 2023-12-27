import argparse
import math
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, Sampler
from typing import List

from utils import gen_points, gen_seeds, read_tensors



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


# -------------------- MODULES -------------------- #

class DistributedStratifiedCovarianceDataset(Dataset):

    # TODO: Do we need each rank to have same number of points?
    #       Is this a requirement of ring all-reduce?

    def __init__(self, dir: str, rank: int, world_size: int):

        # Read in this rank's data and all ranks' stratum counts
        num_strata = 2 * world_size + 1
        strat_counts = torch.zeros(world_size, num_strata, dtype=torch.int32)
        for r in range(world_size):
            strat_ = torch.load(os.path.join(dir, f'stratum-{r}.pt'))
            strat_counts[r, :] = torch.bincount(strat_)

            if r == rank:
                strat = strat_
                points = torch.load(os.path.join(dir, f'points-{r}.pt'))
                cov = torch.load(os.path.join(dir, f'cov-{r}.pt'))

        # Create dictionaries that map stratum to points/cov
        self.points = None
        self.cov = None
        self.num_iters = None
        self.strat_points = {}
        self.strat_cov = {}
        self.strat_num_iters = {}
        for s in range(num_strata):
            mask = strat == s
            self.strat_points[s] = points[mask]
            self.strat_cov[s] = cov[mask]
            self.strat_num_iters[s] = strat_counts[:,s].max().item()

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def set_stratum(self, stratum):
        self.points = self.strat_points[stratum]
        self.cov = self.strat_cov[stratum]
        self.num_iters = self.strat_num_iters[stratum]

    def get_num_iters(self):
        return self.num_iters
    

class DistributedStratifiedDatasetSampler(Sampler):

    def __init__(
            self, 
            dataset: DistributedStratifiedCovarianceDataset, 
            gen: torch.Generator
        ) -> None:
        self.dataset = dataset
        self.gen = gen

    def __iter__(self):
        idx = torch.randperm(len(self.dataset), generator=self.gen)
        pad_size = self.dataset.get_num_iters() - len(idx)
        idx_pad = torch.randperm(len(idx), generator=self.gen)[:pad_size]
        return iter(idx.tolist() + idx_pad.tolist())
    

# NOTE: Should this inherit from DataLoader and implement a set_stratum method?
class StratifiedDataLoader(object):

    def __init__(
            self, 
            dataset: DistributedStratifiedCovarianceDataset, 
            sampler: DistributedStratifiedDatasetSampler, 
            batch_size: int
        ) -> None:
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

    def set_stratum(self, stratum):
        self.dataset.set_stratum(stratum)


# TODO: Move sync_model functionality into this class
class LowRankCovariance(nn.Module):

    def __init__(self, num_vars: int, num_facs: int, gen: torch.Generator):
        super().__init__()
        self.num_facs = num_facs
        self.loads = nn.Embedding(num_vars, num_facs, dtype=torch.float64)  # TODO: Custom initialization (SVD?)
        self.loads.weight.data = torch.randn(num_vars, num_facs, generator=gen, dtype=torch.float64)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)  # NOTE: Has length len(idx0)
    
    def get_loads(self, idx=None):
        if idx is None:
            return self.loads.weight.data
        else:
            return self.loads.weight.data[idx]
   
    def set_loads(self, loads, idx=None):
        if idx is None:
            self.loads.weight.data = loads
        else:
            self.loads.weight.data[idx] = loads
    



# -------------------- DISTRIBUTION -------------------- #

def init_process(rank, world_size, dir_data, num_vars, num_facs, lr, max_epochs, seeds, fcn, backend):
    path = '/tmp/sharedfile'
    if os.path.exists(path):
        os.remove(path)
    dist.init_process_group(
        backend, init_method=f'file://{path}',
        rank=rank, world_size=world_size
    )
    fcn(rank, world_size, dir_data, num_vars, num_facs, lr, max_epochs, seeds)


# -------------------- RUN -------------------- #

def fair_allocate(num_items: int, num_groups: int) -> List[int]:
    """Evenly distributes num_items across num_groups."""
    out = [num_items // num_groups] * num_groups
    remainder = num_items % num_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out


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



def sync_model(
        rank: int, 
        world_size: int, 
        model: LowRankCovariance, 
        points: torch.Tensor
    ):

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
    idx_in = {r: torch.zeros(sz_in[r].item(), dtype=torch.int32) for r in other_ranks}
    reqs_idx_out = {}
    reqs_idx_in = {}
    loads_in = {r: torch.zeros(sz_in[r].item(), 2, dtype=torch.float64) for r in other_ranks}
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


def broadcast_model(model: LowRankCovariance, rank: int, src: int):
    if rank == src:
        loads = model.get_loads()
    else:
        loads = torch.zeros_like(model.get_loads())
    dist.broadcast(loads, src)
    model.set_loads(loads)


def run(rank, world_size, dir, num_vars, num_facs, lr, max_epochs, seeds):

    num_strata = 2 * world_size + 1
    seed = seeds[rank]
    gen = torch.Generator().manual_seed(seed)

    dataset = DistributedStratifiedCovarianceDataset(dir, rank, world_size)
    sampler = DistributedStratifiedDatasetSampler(dataset, gen)
    dataloader = StratifiedDataLoader(dataset, sampler, batch_size=3)

    model = LowRankCovariance(num_vars, num_facs, gen)
    broadcast_model(model, rank, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(max_epochs):  # > epoch
        
        # Broadcast stratum sequence from rank 0
        if rank == 0:
            strat_seq = torch.randperm(num_strata, dtype=torch.int32)
        else:
            strat_seq = torch.zeros(num_strata, dtype=torch.int32)
        dist.broadcast(strat_seq, 0)

        for s in strat_seq:  # > subepoch

            dataloader.set_stratum(s.item())

            for points, cov in dataloader:

                # Forward pass
                preds = model(points[:,0], points[:,1])
                loss = mse_loss(preds, cov)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Sync model
                dist.barrier()
                sync_model(rank, world_size, model, points)
                dist.barrier()

            # (1) TODO: Sync model after each step
            #           - This hangs because ranks have different number of points
            # (2) TODO: Project
            # (3) TODO: Evaluate loss after each epoch

        if rank == 0:
            print(f"epoch = {epoch} | loss = loss")


    dist.destroy_process_group()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='gloo')
    parser.add_argument('-ws', '--world_size', type=int)
    args = parser.parse_args()

    # Configure globals
    dir_data = './data/ffa-dist/data'
    dir_cov = './data/ffa-dist/cov'
    delta = 0.1
    num_facs = 2
    lr = 0.01
    max_epochs = 50
    seed = 12345
    gen = torch.Generator().manual_seed(seed)
    seeds = gen_seeds(gen, args.world_size)

    # Delete files from covariance directory
    if os.path.exists(dir_cov):
        shutil.rmtree(dir_cov)
    os.makedirs(dir_cov)


    # ---------- POINT STRATIFICATION AND ALLOCATION ---------- #

    # Get the number of variables
    path = os.path.join(dir_data, 'data-0.pt')
    num_vars = torch.load(path).shape[0]

    # Generate strata and a dict that maps blocks to their rank and stratum number
    strata = gen_strata(args.world_size)
    blocks_map = {}
    i = 0
    while i < len(strata):
        stratum = strata[i]
        rank = 0
        for block in stratum:
            blocks_map[block] = {'rank': rank, 'stratum': i}
            rank += 1
        i += 1

    # Segment variables
    seg_cnts = fair_allocate(num_vars, 2*args.world_size)
    idx = 0
    segs = torch.zeros(num_vars, dtype=torch.int32)
    for seg, cnt in enumerate(seg_cnts):
        segs[idx:(idx+cnt)] = torch.ones(cnt) * seg
        idx += cnt
    segs = segs[torch.randperm(num_vars, generator=gen)]

    # For worker i (i = 0, ..., world_size - 1):
    #   - Create points-{i}.pt file of points
    #   - Create stratum-{i}.pt file of strata
    points_loader = gen_points(num_vars, delta, 50)
    for points_batch in points_loader:
        sz = len(points_batch)

        seg0_batch = segs[points_batch[:,0]]
        seg1_batch = segs[points_batch[:,1]]

        rank_batch = torch.zeros(sz, dtype=torch.int32)
        stratum_batch = torch.zeros(sz, dtype=torch.int32)

        for i in range(sz):

            block = tuple(sorted(
                [seg0_batch[i].item(), seg1_batch[i].item()], 
                reverse=True
            ))
            rank_batch[i] = blocks_map[block]['rank']
            stratum_batch[i] = blocks_map[block]['stratum']
        
        for r in range(args.world_size):

            mask = rank_batch == r
            points_rank = points_batch[mask]
            stratum_rank = stratum_batch[mask]

            path_points = os.path.join(dir_cov, f'points-{r}.pt')
            path_stratum = os.path.join(dir_cov, f'stratum-{r}.pt')

            if os.path.exists(path_points):
                old_points_rank = torch.load(path_points)
                old_stratum_rank = torch.load(path_stratum)
                points_rank = torch.cat((old_points_rank, points_rank))
                stratum_rank = torch.cat((old_stratum_rank, stratum_rank))

            torch.save(points_rank, path_points)
            torch.save(stratum_rank, path_stratum)



    # ---------- COVARIANCE PREPARATION ---------- #

    # Compute covariance for each rank's points
    # TODO: Parallelize

    for r in range(args.world_size):
        path_points = os.path.join(dir_cov, f'points-{r}.pt')
        points = torch.load(path_points)
        num_points = len(points)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        dataloader = read_tensors(dir_data, 'data')
        for batch in dataloader: 
            n += batch.shape[1]
            for i in range(num_points):
                row, col = points[i, :]
                t1[i] += torch.sum(batch[row,:] * batch[col,:])
                t2[i] += torch.sum(batch[row,:])
                t3[i] += torch.sum(batch[col,:])
            cov = (t1 - t2 * t3 / n) / (n - 1)
            path_cov = os.path.join(dir_cov, f'cov-{r}.pt')
            torch.save(cov, path_cov)


    # ---------- DISTRIBUTED RUN ---------- #

    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, 
                dir_cov, num_vars, num_facs, lr, max_epochs, seeds,
                run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()