import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from typing import List

from utils import (
    create_second_difference_matrix, 
    gen_points, 
    gen_seeds, 
    read_tensors,
    refresh_directory
)



# -------------------- UTILITIES -------------------- #

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
        dir_cov,
        dir_model,
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
        rank, world_size, dir_cov, dir_model,
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
        dir_cov: str,
        dir_model: str,
        num_vars: int, 
        num_facs: int, 
        alpha: float,
        lr: float, 
        max_epochs: int, 
        seeds: List[int]
    ) -> None:
    
    seed = seeds[rank]
    gen = torch.Generator().manual_seed(seed)

    dataset = DistributedCovarianceDataset(dir_cov, rank, world_size)
    print(f"rank {rank} | len(dataset) = {len(dataset)}")
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
        path = os.path.join(dir_model, 'cov-model.pth')
        state_dict = model.state_dict()
        state_dict['loads.weight'] = state_dict.pop('module.loads.weight')  # Replace DDP key
        torch.save(state_dict, path)

    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='gloo')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--num_facs', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # Configure directories
    dir_data = os.path.join('.', 'data', args.dir, 'data')
    dir_cov = os.path.join('.', 'data', args.dir, 'cov')
    dir_model = os.path.join('.', 'data', args.dir, 'model')
    
    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Delete files from covariance and model directories
    refresh_directory(dir_cov)
    refresh_directory(dir_model)

    
    # ---------- COVARIANCE PREPARATION ---------- $

    print(f"Computing covariance...")

    # Get the number of variables
    path = os.path.join(dir_data, 'data-0.pt')
    num_vars = torch.load(path).shape[0]

    # Generate points then compute covariance
    cov_batch_size_per_proc = int((num_vars ** 2) / 18)  # TODO: Why? Better choice?
    cov_batch_size = cov_batch_size_per_proc * args.world_size
    points = gen_points(num_vars, args.delta, cov_batch_size)
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
        path_points = os.path.join(dir_cov, f'points-{iter}.pt')
        path_cov = os.path.join(dir_cov, f'cov-{iter}.pt')
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
                rank, args.world_size, dir_cov, dir_model,
                num_vars, args.num_facs, args.alpha, args.lr, args.max_epochs, 
                seeds, run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    # ---------- EVALUATION ---------- #
    path = os.path.join(dir_model, 'cov-model.pth')
    final_model = LowRankCovariance(num_vars, args.num_facs, gen)
    final_model.load_state_dict(torch.load(path))
    df = pd.DataFrame(final_model.loads.weight.data.numpy(), columns=['l1', 'l2'])
    sns.lineplot(data=df)
    path = os.path.join('.', 'data', args.dir, 'loads.png')
    plt.savefig(path)


