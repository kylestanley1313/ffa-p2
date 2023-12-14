import os
import math
import shutil
import torch

from utils import write_generated_tensor



def build_loadings(fcns, num_vars):
    """Builds a loading matrix with `num_vars` rows and `len(fcns)` columns 
    from `fcns`."""
    x = torch.linspace(0, 1, num_vars, dtype=torch.float64)
    num_facs = len(fcns)
    loads = torch.zeros(num_vars, num_facs, dtype=torch.float64)
    for k in range(num_facs):
        loads[:,k] = fcns[k](x)
    return loads


def sine_loading(x):
    return torch.sin(2 * math.pi * x)


def cosine_loading(x):
    return torch.cos(2 * math.pi * x)


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
    while n > 0:
        n_batch = min(batch_size, n) 
        facs = torch.normal(0, 1, (num_facs, n_batch), dtype=torch.float64, generator=gen)
        errs = torch.normal(0, 1, (num_vars, n_batch), dtype=torch.float64, generator=gen)
        errs *= err_sds.view(-1, 1)
        data = torch.matmul(loadings, facs) + errs
        yield data
        n -= n_batch


if __name__ == '__main__':

    # Configure globals
    dir_data = './data/ffa'
    num_train = 100
    num_val = 0
    data_batch_size = 50
    num_vars = 30
    load_fcns = [sine_loading, cosine_loading]
    num_facs = len(load_fcns)
    
    # Seeding
    seed = 12345
    gen = torch.Generator().manual_seed(seed)

    # ---------- DATA SIMULATION ---------- #
    print(f"Simulating new data...")

    # Delete files from directory
    shutil.rmtree(dir_data)
    os.makedirs(dir_data)

    loadings = build_loadings(load_fcns, num_vars)
    err_sds = 0.2 * torch.ones(loadings.shape[0], dtype=torch.float64)
    data = simulate_ffm_data(
        loadings, 
        err_sds,
        num_train=num_train,
        num_val=num_val,
        batch_size=data_batch_size,
        gen=gen
    )

    write_generated_tensor(data, dir_data, 'data')

    print("DONE!")
