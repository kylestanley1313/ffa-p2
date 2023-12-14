import os
import math
import torch
from typing import Generator



# -------------------- MISCELLANEOUS -------------------- #

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
    

def write_generated_tensor(tensor: Generator, dir: str, prefix: str):
    for i, batch in enumerate(tensor):
        path = os.path.join(dir, f'{prefix}-{i}.pt')
        torch.save(batch, path)


def read_tensors(dir, prefix):
    files = sorted(os.listdir(dir))
    for f in files:
        if f.startswith(prefix) and f.endswith('.pt'):
            yield torch.load(os.path.join(dir, f))



# -------------------- TRAINING POINTS -------------------- #

def gen_cartesian_prod(grid):
    """Yields elements of the cartesian product grid x grid in batches of
    size len(grid)."""
    grid_size = len(grid)
    for i in range(0, grid_size):
        idx = torch.cartesian_prod(
            torch.tensor([i], dtype=torch.int32),
            torch.arange(grid_size, dtype=torch.int32)
        )
        yield torch.column_stack((grid[idx[:,0]], grid[idx[:,1]]))


def gen_points(num_vars: int, delta: float, batch_size: int) -> Generator:
    """Yields training points for a num_vars-by-num_vars covariance matrix in
    batches."""
    if batch_size < num_vars: 
        raise Exception("Must have batch_size >= num_vars")

    grid = torch.arange(num_vars, dtype=torch.int32)
    bandwidth = math.ceil(num_vars * delta)

    start_new_batch = True
    leftovers = None
    for cp_batch in gen_cartesian_prod(grid): 

        if start_new_batch:
            start_new_batch = False
            batch = torch.zeros(batch_size, 2, dtype=torch.int32)
            start_idx = 0
            num_leftovers = len(leftovers) if leftovers is not None else 0
            if num_leftovers > 0:
                batch[:num_leftovers] = leftovers
                start_idx = num_leftovers
                leftovers = None

        keep = cp_batch[:,1] < cp_batch[:,0] - bandwidth
        cp_batch = cp_batch[keep]
        num_to_keep = len(cp_batch)
        num_to_inc = min(num_to_keep, batch_size - start_idx)
        num_to_exc = max(0, num_to_keep - num_to_inc)
        batch[start_idx:(start_idx + num_to_inc)] = cp_batch[:num_to_inc]
        start_idx += num_to_inc
        
        # Iteration Cases: 
        #  [Any]
        #   (1) cb_batch overfills batch --> start_idx == batch_size and num_to_exc > 0
        #   (2) cb_batch precisely fills batch --> start_idx == batch_size and num_to_exc < 0
        #  [Last]
        #   (3) cb_batch underfills batch --> start_idx < batch_size
        if start_idx == batch_size:  # if (1) or (2), yield saturated batch
            yield batch
            start_new_batch = True
            if num_to_exc > 0:
                leftovers = cp_batch[-num_to_exc:]
    
    if leftovers is not None:  # if (2), yield leftovers
        yield leftovers
    elif start_idx < batch_size:  # if (3), yield underfilled batch
        yield batch[:start_idx]
