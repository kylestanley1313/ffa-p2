import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from sklearn.decomposition import PCA
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Optional, Tuple

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils import (
    flatten_dataset,
    gen_points, 
    gen_seeds, 
    multiply_list,
    read_tensors,
    remove_file,
    refresh_directory,
    write_rows_to_csv
)


# -------------------- MODULES -------------------- #

class DistributedStratifiedCovarianceDataset(Dataset):

    def __init__(self, dir: str, rank: int, world_size: int):

        # Read in this rank's data
        strat = torch.load(os.path.join(dir, f'stratum-{rank}.pt'))
        points = torch.load(os.path.join(dir, f'points-{rank}.pt'))
        cov = torch.load(os.path.join(dir, f'cov-{rank}.pt'))

        # Create dictionaries that map stratum to points/cov
        num_strata = 2 * world_size + 1
        self.points = None
        self.cov = None
        self.strat_points = {}
        self.strat_cov = {}
        for s in range(num_strata):
            mask = strat == s
            self.strat_points[s] = points[mask]
            self.strat_cov[s] = cov[mask]

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, index):
        return self.points[index], self.cov[index]
    
    def set_stratum(self, stratum):
        self.points = self.strat_points[stratum]
        self.cov = self.strat_cov[stratum]

    def set_all_strata(self):
        self.points = torch.cat(list(self.strat_points.values()))
        self.cov = torch.cat(list(self.strat_cov.values()))
    
    def storage(self):
        """Returns size of dataset (in bytes)."""
        self.set_all_strata()
        points_sz = sys.getsizeof(self.points.untyped_storage())
        cov_sz = sys.getsizeof(self.cov.untyped_storage())
        return points_sz + cov_sz
    

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
        return iter(idx.tolist())
    

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

    def __len__(self):
        return len(self.dataset)

    def set_stratum(self, stratum):
        self.dataset.set_stratum(stratum)

    def set_all_strata(self):
        self.dataset.set_all_strata()


class LowRankCovariance(nn.Module):

    def __init__(
            self, 
            num_vars: int, 
            num_facs: int, 
            path_init: Optional[str] = None
        ):
        super().__init__()
        self.num_facs = num_facs
        self.loads = nn.Embedding(num_vars, num_facs, dtype=torch.float64)  # TODO: Custom initialization (SVD?)
        if path_init:
            self.loads.weight.data = torch.load(path_init)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)
    
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
    


# -------------------- RUN -------------------- #

def fair_allocate(num_items: int, num_groups: int) -> List[int]:
    """Evenly distributes num_items across num_groups."""
    out = [num_items // num_groups] * num_groups
    remainder = num_items % num_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out


def gen_strata(nprocs: int) -> List[Tuple[Tuple]]:
    path = os.path.join('strata', f'nprocs-{nprocs}.pt')
    if os.path.exists(path):
        tensor = torch.load(path)
        num_strata = 2 * nprocs + 1
        strata = [None] * num_strata
        for s in range(num_strata):
            mask = tensor[:,0] == s
            stratum = tuple(tuple(s_) for s_ in tensor[mask, 1:].tolist())
            strata[s] = stratum
        return strata
    else:
        raise Exception(f"Strata do not exist for {nprocs} processes.")


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
    idx_in = {
        r: torch.zeros(sz_in[r].item(), dtype=torch.int32) 
        for r in other_ranks
    }
    reqs_idx_out = {}
    reqs_idx_in = {}
    loads_in = {
        r: torch.zeros(sz_in[r].item(), model.num_facs, dtype=torch.float64) 
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


def broadcast_model(model: LowRankCovariance, rank: int, src: int):
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
        num_strata, 
        rank, 
        world_size
    ):

    bench = True if dist.get_rank() == 0 else False
    time_model = 0
    time_objective = 0
    time_backward = 0
    time_step = 0

    # Broadcast stratum sequence from rank 0
    if rank == 0:
        strat_seq = torch.randperm(num_strata, generator=gen, dtype=torch.int32)
    else:
        strat_seq = torch.zeros(num_strata, dtype=torch.int32)

    for s in strat_seq:  # > subepoch
        dataloader.set_stratum(s.item())

        for points, cov in dataloader:

            # Forward pass
            start = time.time()
            preds = model(points[:,0], points[:,1])
            end = time.time()
            time_model += end - start
            start = time.time()
            loss = objective(preds, cov)
            end = time.time()
            time_objective += end - start

            # Backward pass
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            end = time.time()
            time_backward += end - start
            start = time.time()
            optimizer.step()
            end = time.time()
            time_step += end - start

        # Sync model
        dist.barrier()
        sync_model(rank, world_size, model, points)

    msg = (
        f"""
        Times: 
            time_model = {time_model}
            time_objective = {time_objective}
            time_backward = {time_backward}
            time_step = {time_step}
        """
    )
    if bench: 
        print(msg)


def compute_loss(model, dataloader, objective, rank, world_size):

    # Compute rank-wise loss
    dataloader.set_all_strata()  # give dataloader access to all a worker's points
    loss = 0
    n = len(dataloader)
    for points, cov in dataloader:
        preds = model(points[:,0], points[:,1])
        loss += objective(preds, cov)
    
    # Send loss data when rank > 0
    if rank != 0:
        dist.send(torch.tensor([n], dtype=torch.int32), 0)
        dist.send(torch.tensor([loss], dtype=torch.float64), 0)
    
    # Receive loss data when rank == 0
    if rank == 0:
        
        # Collect losses
        n_list = [torch.zeros(1, dtype=torch.int32) for _ in range(world_size)]
        loss_list = [torch.zeros(1, dtype=torch.float64) for _ in range(world_size)]
        n_list[0] = torch.tensor([n], dtype=torch.int32)
        loss_list[0] = torch.tensor([loss], dtype=torch.float64)
        for r in range(1, world_size):
            dist.recv(n_list[r], r)
            dist.recv(loss_list[r], r)

        # Aggregate losses
        n = sum(n_list).item()
        loss = sum(loss_list).item()
        return loss / n
    

def init_process(
        rank: int, 
        world_size: int, 
        path_shared: str,
        config: str, 
        dir_out: str, 
        grid_shape: List[str], 
        num_facs: int, 
        batch_size: int,
        lr: float, 
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
        grid_shape, num_facs,  
        batch_size, lr, max_epochs, seed
    )


def run(
        rank: int, 
        world_size: int, 
        config: str,
        dir_out: str,
        grid_shape: List[int], 
        num_facs: int, 
        batch_size: int,
        lr: float, 
        max_epochs: int, 
        seed: int
    ) -> None:

    config = load_config(config)
    gen = torch.Generator().manual_seed(seed)
    num_strata = 2 * world_size + 1

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
        init=DistributedStratifiedCovarianceDataset,
        dir=dir_bench,
        prefix='dataset',
        benchmark=config.benchmark
    )

    dataset = Dataset_(dir_cov, rank, world_size)
    sampler = DistributedStratifiedDatasetSampler(dataset, gen)
    dataloader = StratifiedDataLoader(dataset, sampler, batch_size=batch_size)

    num_vars = multiply_list(grid_shape)
    path_init = os.path.join(dir_out, 'init_loads.pt')
    model = LowRankCovariance(num_vars, num_facs, path_init)
    broadcast_model(model, rank, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    objective_mean = partial(mse_loss, reduction='mean')
    objective_sum = partial(mse_loss, reduction='sum')

    for epoch in range(max_epochs):

        process_epoch_(
            model, dataloader, objective_mean, optimizer, 
            gen, num_strata, rank, world_size
        )
        loss = compute_loss(model, dataloader, objective_sum, rank, world_size)
        if rank == 0:
            print(f"epoch = {epoch} | loss = {loss}")

    # Save model
    if rank == 0:
        path = os.path.join(dir_out, 'cov-model.pth')
        state_dict = model.state_dict()
        torch.save(state_dict, path)

    dist.destroy_process_group()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--num_facs', type=int)
    parser.add_argument('--delta', type=float)
    parser.add_argument(
        '--init_method', type=str, 
        choices=['random', 'pca_full', 'pca_arpack', 'pca_randomized']
    )
    parser.add_argument('--init_perc', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Configure globals
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Set directories and paths
    dir_dataset = os.path.join(config.scratch_root, 'datasets', args.dataset)
    dir_out = os.path.join('out', args.dir_out)
    dir_data = os.path.join(config.scratch_root, dir_out, 'data')
    dir_cov = os.path.join(config.scratch_root, dir_out, 'cov')
    dir_bench = os.path.join(dir_out, 'bench')
    path_init = os.path.join(dir_out, 'init_loads.pt')
    path_model = os.path.join(dir_out, 'cov-model.pth')
    other_bench_path = os.path.join(dir_bench, 'other.csv')

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


    # ---------- POINT STRATIFICATION AND ALLOCATION ---------- #
    print("Stratifying training points and allocating to processes...")

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
    points_loader = gen_points(grid_shape, args.delta, int(1.2 * num_vars))
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
    print("Computing covariance...")

    # Compute covariance for each rank's points
    # TODO: Parallelize
    start = time.time()
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
            n += len(batch)
            for i in range(num_points):
                row, col = points[i, :]
                t1[i] += torch.sum(batch[:,row] * batch[:,col])
                t2[i] += torch.sum(batch[:,row])
                t3[i] += torch.sum(batch[:,col])
            cov = (t1 - t2 * t3 / n) / (n - 1)
            path_cov = os.path.join(dir_cov, f'cov-{r}.pt')
            torch.save(cov, path_cov)
    end = time.time()
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
        init_loads = torch.randn(num_vars, args.num_facs, generator=gen, dtype=torch.float64)
    else:

        # Read in (possibly subsampled) data
        data = []
        dataloader = read_tensors(dir_data, 'data')
        for batch in dataloader:
            n = len(batch)
            num_to_keep = int(n * args.init_perc)
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
    write_rows_to_csv(other_bench_path, [['initialization', end - start]])


    # ---------- DISTRIBUTED RUN ---------- #

    suffix = args.dir_out.split('/')[-1]
    path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')
    remove_file(path_shared)
    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, path_shared, args.config, dir_out,
                grid_shape, args.num_facs,
                args.batch_size, args.lr, args.max_epochs,
                seeds[rank], run, config.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

