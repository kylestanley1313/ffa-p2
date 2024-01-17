import csv
import os
import pandas as pd
import time
import torch.distributed as dist
from functools import partial
from typing import Any, List, Optional


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


def aggregate_benchmarks(
        dir: str, 
        prefix: str, 
        reduction: Optional[str] = None
    ) -> List[List[Any]]:

    files = [f for f in os.listdir(dir) if f.startswith(prefix)]
    dfs = []
    for f in files:
        path = os.path.join(dir, f)
        dfs.append(pd.read_csv(path, header=None))
    
    df = pd.concat(dfs, ignore_index=True)
    if reduction is None:
        return df.values.tolist()
    elif reduction == 'mean':
        return [[f'mean {prefix}', df.mean().item()]]
    else:
        raise Exception(f"Invalid value passed to `reduction`.")
        

                