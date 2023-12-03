import argparse
import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from typing import Generator, Iterator, List, Optional, Tuple


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
        facs = torch.normal(0, 1, (num_facs, n_batch), dtype=torch.float64, generator=gen)
        errs = torch.normal(0, 1, (num_vars, n_batch), dtype=torch.float64, generator=gen)
        errs *= err_sds.view(-1, 1)
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

def init_process(
        rank, 
        world_size, 
        points_dict,  # TEMP
        blocks_dict,  # TEMP
        cov_dict,     # TEMP
        strata,       # TEMP
        num_vars, 
        num_facs, 
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
        rank, world_size, 
        (points_dict[rank], blocks_dict[rank], cov_dict[rank]), strata,  # TEMP
        num_vars, num_facs, lr, max_epochs, seeds
    )


# -------------------- DATA PREP -------------------- #

def fair_allocate(num_items: int, num_groups: int) -> List[int]:
    """Evenly distributes num_items across num_groups."""
    out = [num_items // num_groups] * num_groups
    remainder = num_items % num_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out
 

def gen_train_points(num_vars: int, delta: float) -> torch.Tensor:
    """Generates covariance matrix training points for data of a given 
    num_vars (M) and delta (bandwidth)."""
    grid = torch.arange(num_vars, dtype=torch.int32)
    points = torch.cartesian_prod(grid, grid)
    bandwidth = math.ceil(num_vars * delta)
    keep = points[:,1] < points[:,0] - bandwidth
    return points[keep,:]


def block_train_points(
        points: torch.Tensor, 
        num_vars: int, 
        num_workers: int,
        gen: torch.Generator
    ) -> torch.Tensor:
    """Assigns (lower-triangular) blocks to training points."""
    block_cnts = fair_allocate(num_vars, 2*num_workers)
    idx = 0
    blocks = torch.zeros(num_vars, dtype=torch.int32)
    for b, cnt in enumerate(block_cnts):
        blocks[idx:(idx+cnt)] = torch.ones(cnt) * b
        idx += cnt
    blocks = blocks[torch.randperm(num_vars, generator=gen)]
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
    return points, blocks   # cols: x[0], x[1] | b[0], b[1]

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



# -------------------- MODULES -------------------- #

class LowRankCovariance(nn.Module):

    def __init__(self, num_vars, num_facs):
        super().__init__()
        self.loads = nn.Embedding(num_vars, num_facs)  # TODO: Custom initialization (SVD?)

    def forward(self, idx0, idx1):
        loads0 = self.loads(idx0)
        loads1 = self.loads(idx1)
        return (loads0 * loads1).sum(dim=1)  # NOTE: Has length len(idx0)
    

class BlockwiseSampler(Sampler):

    def __init__(
            self, 
            blocks: torch.Tensor,
            block: Tuple, 
            generator: torch.Generator
    ) -> None:
        all_idx = torch.arange(len(blocks))
        mask = blocks == torch.tensor(block)
        mask = torch.logical_and(mask[:,0], mask[:,1])
        self.idx = all_idx[mask]
        self.gen = generator

    def __iter__(self) -> Iterator:
        i = torch.randperm(len(self.idx), generator=self.gen).tolist()
        return iter(self.idx[i].tolist())
    
    def __len__(self) -> int:
        return len(self.idx)
    



# -------------------- RUN -------------------- #


def run(
        rank: int, 
        world_size: int, 
        data: Tuple[torch.Tensor], 
        strata: List[Tuple[Tuple]], 
        num_vars: int, 
        num_facs: int, 
        lr: float, 
        max_epochs: int, 
        seeds: List[int]
    ) -> None:
    
    seed = seeds[rank]
    gen = torch.Generator().manual_seed(seed)

    points, blocks, cov = data
    dataset = TensorDataset(points, blocks, cov)
    blocks_unique = [s[rank] for s in strata]
    samplers = {b: BlockwiseSampler(blocks, b, gen) for b in blocks_unique}
    dataloaders = {
        b: DataLoader(dataset, batch_size=1, sampler=samplers[b]) 
        for b in blocks_unique
    }

    model = LowRankCovariance(num_vars, num_facs)
    model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend', default='gloo')
    parser.add_argument('-ws', '--world_size', type=int)
    parser.add_argument('-sn', '--simulate_new', action = 'store_true')
    args = parser.parse_args()

    # Configure globals
    dir_data = './data/ffa'
    delta = 0.1
    num_facs = 2
    lr = 0.01
    max_epochs = 20
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
        ], dtype=torch.float64)
        # loadings = torch.randn(100, 3, dtype=torch.float64, generator=gen)
        err_sds = 0.2 * torch.ones(loadings.shape[0], dtype=torch.float64)
        data = simulate_ffm_data(
            loadings, 
            err_sds,
            num_train=200,
            num_val=100,
            batch_size=50,
            gen=gen
        )
        write_data(data, dir_data)

        print("DONE!")
    else: 
        print("Using existing simulated data.")

    
    # ---------- DATA PREPARATION ---------- $

    # NOTE: This data preparation scheme is temporary. 

    prep_seed = gen_seeds(gen, 1)
    gen = torch.Generator().manual_seed(prep_seed)
    path = os.path.join(dir_data, 'data-0.pt')
    num_vars = torch.load(path).shape[0]
    num_workers = args.world_size

    # Generate points, block points, then compute covariances
    points = gen_train_points(num_vars, delta)
    points, blocks = block_train_points(points, num_vars, num_workers, gen)
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

    # Allocate data to workers
    strata = gen_strata(num_workers)
    points_dict = {}
    blocks_dict = {}
    cov_dict = {}
    for r in range(num_workers):
        worker_blocks = [s[r - 1] for s in strata]
        mask = torch.zeros(len(blocks), dtype=torch.bool)
        for b in worker_blocks: 
            temp = blocks == torch.tensor(b)
            temp = torch.logical_and(temp[:,0], temp[:,1])
            mask = torch.logical_or(mask, temp)
        points_dict[r] = points[mask]
        blocks_dict[r] = blocks[mask]
        cov_dict[r] = cov[mask]


    # ---------- DISTRIBUTED RUN ---------- #

    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, 
                points_dict, blocks_dict, cov_dict, strata,
                num_vars, num_facs, lr, max_epochs, seeds,
                run, args.backend
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()