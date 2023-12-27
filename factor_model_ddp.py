"""Implements DSGD for CMF using the standard DistributedSampler/DDP framework.
This framework contains features we do not want/need: 
    - Gradient synchronization across processes (not needed when data stratified)
    - Dropping or adding data to ensure each process has same number of forward
      and backward calls. 

Since this framework contains more communication (gradient synchronization) than
we require in CMF, we can use it to get an upper limit on compute time. 

Hopefully, we can steal ideas from DDP (like ring all-reduce) to speed up our
computation. 

Also, projection onto smoother loading space may be possible in this framework.

NOTE: (DistributedSampler Data Dropping/Adding)
    - DistributedSampler drops/adds data so that each process has the same
      number of iterations. This ensures that DistributedDataParallel's
      all-reduce step doesn't hang forever. 
    - Relevant Issue: https://github.com/pytorch/pytorch/issues/22584
    - Possible Solution: https://discuss.pytorch.org/t/distributedsampler/90205
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from typing import List

from utils import gen_seeds, gen_points, read_tensors



# -------------------- UTILITIES -------------------- #

def create_second_difference_matrix(n):

    num_idx = 3 * n - 2
    idx = torch.zeros(2, num_idx, dtype=torch.int32)
    vals = torch.zeros(num_idx, dtype=torch.float64)
    cnt = 0
    for i in range(n):  # loop thru rows
        
        # Add diagonal
        idx[:, cnt] = torch.tensor([i, i])
        vals[cnt] = 2
        cnt += 1
        
        if i == 0:
            
            # Add right
            idx[:, cnt] = torch.tensor([i, i + 1])
            vals[cnt] = -1
            cnt += 1

        elif i == n - 1:

            # Add left
            idx[:, cnt] = torch.tensor([i, i - 1])
            vals[cnt] = -1
            cnt += 1

        else: 

            # Add right
            idx[:, cnt] = torch.tensor([i, i + 1])
            vals[cnt] = -1
            cnt += 1

            # Add left
            idx[:, cnt] = torch.tensor([i, i - 1])
            vals[cnt] = -1
            cnt += 1

    diff_mat = torch.sparse_coo_tensor(indices=idx, values=vals, size=[n, n])
    return diff_mat


def roughness_penalty(loads: torch.Tensor, diff_mat: torch.Tensor):
    return torch.trace(loads.t() @ diff_mat @ loads)


def objective(
        preds: torch.Tensor, 
        cov: torch.Tensor, 
        model: 'LowRankCovariance', 
        alpha: float, 
        diff_mat: torch.Tensor
    ):
    loads = list(model.parameters())[0]
    err = mse_loss(preds, cov)
    if alpha > 0:
        pen = roughness_penalty(loads, diff_mat)
        return err + alpha * pen
    else: 
        return err


# -------------------- DISTRIBUTION -------------------- #

def init_process(
        rank, 
        world_size, 
        dir_data,
        num_vars, 
        num_facs, 
        alpha,
        lr, 
        max_epochs, 
        seeds, 
        fcn, 
        backend
    ):
    path = '/tmp/sharedfile'
    if os.path.exists(path):
        os.remove(path)
    dist.init_process_group(
        backend, init_method=f'file://{path}',
        rank=rank, world_size=world_size
    )
    fcn(
        rank, world_size, dir_data,
        num_vars, num_facs, alpha, lr, max_epochs, seeds
    )


# -------------------- MODULES -------------------- #

class LowRankCovariance(nn.Module):

    def __init__(self, num_vars: int, num_facs: int, gen: torch.Generator):
        super().__init__()
        self.loads = nn.Embedding(num_vars, num_facs, dtype=torch.float64)  # TODO: Custom initialization (SVD?)
        self.loads.weight.data = torch.randn(num_vars, num_facs, generator=gen, dtype=torch.float64)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)  # NOTE: Has length len(idx0)


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


def run(
        rank: int, 
        world_size: int, 
        dir: str,
        num_vars: int, 
        num_facs: int, 
        alpha: float,
        lr: float, 
        max_epochs: int, 
        seeds: List[int]
    ) -> None:
    
    seed = seeds[rank]
    print(f"rank = {rank} | seed = {seed}")
    gen = torch.Generator().manual_seed(seed)

    dataset = DistributedCovarianceDataset(dir, rank, world_size)
    sampler = DistributedDatasetSampler(dataset, gen)
    dataloader = BasicDataLoader(dataset, batch_size=3, sampler=sampler)

    model = LowRankCovariance(num_vars, num_facs, gen)
    model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    diff_mat = create_second_difference_matrix(num_vars)

    for epoch in range(max_epochs):

        for points, cov in dataloader:

            # Forward pass
            preds = model(points[:,0], points[:,1])
            loss = objective(preds, cov, model, alpha, diff_mat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute and communicate loss
        preds = model(dataset.points[:,0], dataset.points[:,1])
        loss = objective(preds, dataset.cov, model, alpha, diff_mat)
        if rank > 0: 
            dist.send(loss, 0)
        else: 
            for r in range(1, world_size):
                worker_loss = torch.zeros(1, dtype=torch.float64)
                dist.recv(worker_loss, r)
                loss += worker_loss.item()
            print(f"epoch {epoch} | loss = {loss}")

    # Save model
    if rank == 0:
        path = os.path.join(dir, 'cov-model.pth')
        state_dict = model.state_dict()
        state_dict['loads.weight'] = state_dict.pop('module.loads.weight')  # Replace DDP key
        torch.save(state_dict, path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='gloo')
    parser.add_argument('-ws', '--world_size', type=int)
    args = parser.parse_args()

    # Configure globals
    dir_data = './data/ffa'
    num_vars = 30
    num_facs = 2
    alpha = 0.1
    cov_batch_size_per_proc = int((num_vars ** 2) / 18)
    delta = 0.1
    lr = 0.01
    max_epochs = 200
    
    # Seeding
    seed = 12345
    gen = torch.Generator().manual_seed(seed)
    seeds = gen_seeds(gen, args.world_size)

    
    # ---------- COVARIANCE PREPARATION ---------- $

    print(f"Computing covariance...")

    # Generate points then compute covariance
    cov_batch_size = cov_batch_size_per_proc * args.world_size
    points = gen_points(num_vars, delta, cov_batch_size)
    iter = 0
    for points_batch in points: 
        num_points = len(points_batch)
        t1 = torch.zeros(num_points, dtype=torch.float64)
        t2 = torch.zeros(num_points, dtype=torch.float64)
        t3 = torch.zeros(num_points, dtype=torch.float64)
        n = 0
        data = read_tensors(dir_data, 'data')
        for data_batch in data: 
            n += data_batch.shape[1]
            for i in range(num_points): 
                row, col = points_batch[i,:]
                t1[i] += torch.sum(data_batch[row,:] * data_batch[col,:])
                t2[i] += torch.sum(data_batch[row,:])
                t3[i] += torch.sum(data_batch[col,:])
        cov = (t1 - t2 * t3 / n) / (n - 1)
        path_points = os.path.join(dir_data, f'points-{iter}.pt')
        path_cov = os.path.join(dir_data, f'cov-{iter}.pt')
        torch.save(points_batch, path_points)
        torch.save(cov, path_cov)
        iter += 1

    # ---------- DISTRIBUTED RUN ---------- #

    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, dir_data,
                num_vars, num_facs, alpha, lr, max_epochs, seeds,
                run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    # ---------- EVALUATION ---------- #
    path = os.path.join(dir_data, 'cov-model.pth')
    final_model = LowRankCovariance(num_vars, num_facs, gen)
    final_model.load_state_dict(torch.load(path))
    df = pd.DataFrame(final_model.loads.weight.data.numpy(), columns=['l1', 'l2'])
    sns.lineplot(data=df)
    plt.show()


