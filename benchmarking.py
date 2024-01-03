import csv
import os
import pandas as pd
import time
import torch.distributed as dist
from functools import partial


def time_dist_fcn(fcn, dir, prefix, benchmark):
    """Decorator that records the execution times of serial calls to `fcn`."""

    def wrapper(*args, **kwargs):

        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()

        rank = dist.get_rank()
        path = os.path.join(dir, f'{prefix}-{rank}.csv')
        mode = 'a' if os.path.exists(path) else 'w'
        with open(path, mode, newline='') as file:
            writer = csv.writer(file)
            writer.writerow([end - start])
        
        return result

    return wrapper if benchmark else fcn


def size_dist_obj(init, dir, prefix, benchmark):
    """Decorator that records the size of an object initialized with `init`."""

    def wrapper(*args, **kwargs):

        obj = init(*args, **kwargs)

        rank = dist.get_rank()
        path = os.path.join(dir, f'{prefix}-{rank}.csv')
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([obj.storage()])

        return obj
    
    return wrapper if benchmark else init


def aggregate_benchmarks(dir, prefix, world_size, reduction):

    dfs = []
    for r in range(world_size):
        path = os.path.join(dir, f'{prefix}-{r}.csv')
        dfs.append(pd.read_csv(path, header=None))
    
    if reduction == 'mean':
        df = pd.concat(dfs, ignore_index=True)
        out = round(df.mean().item(), 4)
        print(f"Mean {prefix}: {out}")
    else:
        raise Exception(f"Invalid value passed to `reduction`.")
        

                