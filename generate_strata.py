import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple

from utils import gen_seeds

# NOTE: (Ideas)
#   - If finding non-symmetric strata is easy and symmetric strata are a subset
#     of non-symmetric strata, then generate the former set then search for the
#     latter in the generated set. 
#   - Deterministic algorithm.


class ImpossibleStratum(Exception):
    pass


class StrataAlreadyFound(Exception):
    pass


def valid_strata(num_procs: int, strata: List[List[Tuple]]) -> bool:
    """Checks to see whether `strata` is valid. 

    NOTE: Creates an upper triangular matrix (diagonal excluded) called
          `blocks` that contains 1s above the diagonal and zeros elsewhere.
          This function then loops through the blocks in `strata` and adds 1 
          to `blocks[block]` whenever `block` is encountered. If `strata` is
          valid, then `blocks` should be a matrix of ones at the end. 

    Args:
        num_procs (int): Number of processes for which `strata` was created.
        strata (List[List[Tuple]]): Proposed strata

    Returns:
        bool: True if valid; False otherwise.
    """

    # Create `blocks` to be used in off-diagonal strata selection
    sz = 2 * num_procs
    blocks = torch.zeros(sz, sz, dtype=torch.int32)
    for i in range(sz):
        for j in range(i + 1, sz):
                blocks[i,j] = 1
    
    # Update `blocks` with `strata`
    for stratum in strata:
        for block in stratum:
            blocks[block] += 1

    ones = torch.ones(sz, sz, dtype=torch.int32)
    if torch.equal(blocks, ones):
        return True
    else:
        return False
    

def choose_off_diag_stratum(blocks: torch.Tensor, gen: torch.Generator, path: str) -> List[Tuple]:
    """Chooses a stratum given the entries in `blocks`.

    Args:
        blocks (torch.Tensor): Tensor describing the state of stratum selection. 
        gen (torch.Generator): Torch generator.
        path (str): Path to eventual strata file.

    Returns:
        List[Tuple]: Chosen stratum. If it is not possible to choose a stratum, 
            then raise an ImpossibleStratum exception. If a strata file already
            exists at `path`, then raise a StrataAlreadyFound exception.
    """


    def set_ineligible_blocks(blocks, block):
        row = block[0]
        col = block[1]
        blocks[row, blocks[row, :] == 0] = 2
        blocks[blocks[:, col] == 0, col] = 2
        blocks[col, blocks[col, :] == 0] = 2
        blocks[blocks[:, row] == 0, row] = 2
    
    blocks = blocks.clone()
    sz = blocks.shape[0]
    num_procs = int(sz / 2)
    stratum = [None] * num_procs
    attempted_blocks = [torch.zeros(sz, sz, dtype=torch.int32) for _ in range(num_procs)]
    
    r = 0
    while r < num_procs:

        # If strata exists, raise StrataAlradyFound exception
        if os.path.exists(path):
            raise StrataAlreadyFound

        # Get eligible blocks
        zero_idx_1 = torch.nonzero(attempted_blocks[r] == 0).tolist()
        zero_idx_2 = torch.nonzero(blocks == 0).tolist()
        candidates = list(
            set(map(tuple, zero_idx_1)) & 
            set(map(tuple, zero_idx_2))
        )

        if len(candidates) < num_procs - r:

            if r == 0:
                raise ImpossibleStratum
            
            else:  # roll back to previous worker

                attempted_blocks[r] = torch.zeros(sz, sz, dtype=torch.int32)
                r -= 1
                bad_stratum = stratum[r]
                stratum[r] = None
                attempted_blocks[r][bad_stratum] = -1

                # Reset `blocks`
                blocks[bad_stratum] = 0
                blocks[blocks == 2] = 0
                for block in stratum:
                    if block:
                        set_ineligible_blocks(blocks, block)

                continue
        
        # Choose a candidate and add to stratum
        rand_idx = torch.randint(len(candidates), (1,), generator=gen).item()
        block = candidates[rand_idx]
        stratum[r] = block

        # Update `blocks` and `attempted_blocks`
        set_ineligible_blocks(blocks, block)
        blocks[block] = 1
        attempted_blocks[r][block] = -1

        r += 1

    return stratum


def gen_strata(num_procs: int, gen: torch.Generator, path: str) -> List[List[Tuple]]:
    """Generates strata for `num_procs`.

    Args:
        num_procs (int): Number of processes.
        gen (torch.Generator): Torch generator.
        path (str): Path to eventual strata file.

    Returns:
        List[List[Tuple]]: Generated strata.
    """

    # Add diagonal strata
    d1 = tuple((r, r) for r in range(num_procs))
    d2 = tuple((r, r) for r in range(num_procs, 2 * num_procs))
    strata = [d1, d2]

    # Create `blocks` to be used in off-diagonal strata selection
    sz = 2 * num_procs
    blocks = torch.zeros(sz, sz, dtype=torch.int32)
    for i in range(sz):
        for j in range(i, sz):
                blocks[i,j] = -1

    num_strata = 2 * num_procs + 1
    while len(strata) < num_strata:
        
        try:
            stratum = choose_off_diag_stratum(blocks, gen, path)
            strata.append(stratum)
            for block in stratum:
                blocks[block] = -1

        except ImpossibleStratum:

            # Roll back prior stratum selection
            bad_stratum = strata[-1]
            strata = strata[:-1]
            for block in bad_stratum:
                blocks[block] = 0

        except StrataAlreadyFound:
            return None
        
    return strata


def write_strata(strata: List[Tuple[Tuple]], path: str) -> None:
    tensors = []
    for s in range(len(strata)):
        strat_num = s * torch.ones(len(strata[0]), 1, dtype=torch.int32)
        blocks = torch.tensor(strata[s], dtype=torch.int32)
        tensor = torch.cat((strat_num, blocks), dim=1)
        tensors.append(tensor)
    out = torch.cat(tensors)
    torch.save(out, path)


def init_process(rank, world_size, backend, shared_path, fcn, num_procs, seed, path):
    dist.init_process_group(
        backend, init_method=f'file://{shared_path}',
        rank=rank, world_size=world_size
    )
    fcn(rank, num_procs, seed, path)


def run(rank, num_procs, seed, path):

    gen = torch.Generator().manual_seed(seed)
    strata = gen_strata(num_procs, gen, path)
    if not strata: 
        print(f"Rank {rank} did not find strata.")
    elif not valid_strata(num_procs, strata):
        print(f"Rank {rank} produced invalid strata!")
    else: 
        print(f"Rank {rank} produced valid strata.")
        write_strata(strata, path)



if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='gloo')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--num_procs', type=int)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # Seeding
    gen = torch.Generator().manual_seed(args.seed)
    seeds = gen_seeds(gen, args.world_size)

    # Remove existing file
    path = os.path.join('.', 'strata', f'nprocs-{args.num_procs}.pt')
    if os.path.exists(path):
        os.remove(path)

    # Distributed stratum generation
    shared_path = '/tmp/sharedfile'
    if os.path.exists(shared_path):
        os.remove(shared_path)
    processes = []
    mp.set_start_method('spawn')
    for rank in range(args.world_size):
        p = mp.Process(
            target=init_process, 
            args=(
                rank, args.world_size, args.backend, shared_path,
                run, args.num_procs, seeds[rank], path
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

