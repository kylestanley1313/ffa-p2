import argparse
import os
import torch
import torch.multiprocessing as mp
from typing import List, Tuple

from config import load_config
from utils import (
    gen_tensors,
    init_process,
    load_yaml,
    refresh_directory,
    remove_file,
)


def fair_allocate(n_items: int, n_groups: int) -> List[int]:
    """Evenly distributes n_items across n_groups."""
    out = [n_items // n_groups] * n_groups
    remainder = n_items % n_groups
    out[0:remainder] = [x + 1 for x in out[0:remainder]]
    return out


def gen_strata(nprocs: int) -> List[Tuple[Tuple]]:
    path = os.path.join('strata', f'nprocs-{nprocs}.pt')
    if os.path.exists(path):
        tensor = torch.load(path)
        n_strata = 2 * nprocs + 1
        strata = [None] * n_strata
        for s in range(n_strata):
            mask = tensor[:,0] == s
            stratum = tuple(tuple(s_) for s_ in tensor[mask, 1:].tolist())
            strata[s] = stratum
        return strata
    else:
        raise Exception(f"Strata do not exist for {nprocs} processes.")


def allocate_points_ddp(
        rank: int,
        world_size: int,
        dir_cov: str,
        dir_idx: str,
        seed: int,
    ) -> None:
    """Generates and writes files of the form idx-{rank}.pt."""

    gen = torch.Generator().manual_seed(seed)
    points_loader = gen_tensors(dir_cov, 'points_', sort_by=('i', int))
    
    idx_list = []
    n_batch = 0
    start = 0
    for points in points_loader: 

        # Get rank's indices using generator common to all ranks
        sz = len(points)
        start_ = (n_batch + rank) % world_size
        idx = torch.arange(start_, sz, world_size)
        idx = torch.randperm(sz, generator=gen)[idx]
        idx = idx + start

        idx_list.append(idx)
        n_batch += 1
        start += sz

    path = os.path.join(dir_idx, f'idx-{rank}.pt')
    torch.save(torch.cat(idx_list), path)


def allocate_points_dsgd(
        rank: int,
        world_size: int,
        dir_cov: str,
        dir_idx: str,
        n_vars: int, 
        seed: int,
    ) -> None:
    """Generates and writes the files of the form idx-{rank}.pt 
    and strat-{rank}.pt."""    
    gen = torch.Generator().manual_seed(seed)

    # Create dict mapping this rank's blocks to their stratum
    block_map = {}
    strata = gen_strata(world_size)
    for i in range(len(strata)):
        block_map[strata[i][rank]] = i

    # Segment variables
    seg_cnts = fair_allocate(n_vars, 2*world_size)
    idx = 0
    segs = torch.zeros(n_vars, dtype=torch.int32)
    for seg, cnt in enumerate(seg_cnts):
        segs[idx:(idx+cnt)] = torch.ones(cnt) * seg
        idx += cnt
    segs = segs[torch.randperm(n_vars, generator=gen)]

    # Create points and strat files for this rank
    points_loader = gen_tensors(dir_cov, 'points_', sort_by=('i', int))
    idx_list = []
    strat_list = []
    start = 0
    for points in points_loader:

        # Get the segment of each point
        sz = len(points)
        seg0 = segs[points[:,0]]
        seg1 = segs[points[:,1]]

        # Use strat to exclude points not assigned to rank (-1) and collect
        # strata of those assigned to rank (>= 0). 
        strat = -1 * torch.ones(sz, dtype=torch.int32)
        for i in range(sz):
            block = tuple(sorted(
                [seg0[i].item(), seg1[i].item()], 
                reverse=True
            ))
            strat_ = block_map.get(block)
            if strat_ is not None:
                strat[i] = strat_
        mask = strat != -1
        idx = torch.nonzero(mask).squeeze() + start

        idx_list.append(idx.to(torch.int32))
        strat_list.append(strat[mask])
        start += sz

    path_idx = os.path.join(dir_idx, f'idx-{rank}.pt')
    path_strat = os.path.join(dir_idx, f'strat-{rank}.pt')
    torch.save(torch.cat(idx_list), path_idx)
    torch.save(torch.cat(strat_list), path_strat)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--method', type=str, choices=['ddp', 'dsgd'])
    parser.add_argument('--n_procs', type=int)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    config = load_config(args.config)

    # Load design
    path_design = os.path.join(config.group_root, 'designs-lrcmc', f'{args.design}.yml')
    design = load_yaml(path_design)

    # Set directories and paths
    dir_out = os.path.join(config.group_root, 'out-lrcmc', args.design)
    dir_cov = os.path.join(dir_out, 'cov')
    dir_idx = os.path.join(dir_out, f'idx-{args.method}-{args.n_procs}')
    refresh_directory(dir_idx)
    path_shared = os.path.join(config.dir_shared, f'{args.design}-{args.n_procs}')

    
    # ---------- POINT ALLOCATION ---------- #
    allocate_points_fcns = {
        'ddp': allocate_points_ddp,
        'dsgd': allocate_points_dsgd,
    }

    remove_file(path_shared)
    processes = []
    for rank in range(args.n_procs):
        kwargs = {
            'dir_cov': dir_cov,
            'dir_idx': dir_idx,
            'seed': args.seed,  # use same seed for all ranks
        }
        if args.method == 'dsgd':
            kwargs['n_vars'] = design['n_vars']
        p = mp.Process(
            target=init_process,
            args=(
                rank, args.n_procs, allocate_points_fcns[args.method], 
                path_shared, config.backend
            ),
            kwargs=kwargs
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()



