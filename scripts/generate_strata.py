import argparse
import os
import torch

from typing import List, Tuple



def validate_rounds(nprocs: int, rounds: List[List[Tuple]]) -> None: 

    # Generate reference segments and cells
    segs = set(range(1, 2*nprocs))
    cells = set()
    for i in range(2*nprocs):
        for j in range(i + 1, 2*nprocs):
            cells.add((j, i))

    # Check that all segments are in each strata
    for s, round in enumerate(rounds): 
        segs_ = set()
        for i, j in round: 
            segs_.add(i)
            segs_.add(j)
            cells.remove((i, j))

        if len(segs.difference(segs_)) > 0: 
            print(f"Not all segments represented in stratum {s}!")
            return
    
    if len(cells) > 0: 
        print(f"Not all cells represented!")
        return
    
    print(f"Valid strata!")


def factorize_pairs(r):
    """
    Partition the set of all pairs {(i, j): 1 <= i < j <= 2r}
    into 2r-1 subsets (rounds) of r pairs each (a 1-factorization).
    
    Each subset (round) is a list of pairs such that every index 
    in {1,2,...,2r} appears exactly once.
    
    Parameters:
        r (int): half the number of vertices (with total vertices = 2*r).
        
    Returns:
        list of list of tuple: a list containing 2r-1 rounds, each round is a list of r pairs.
    """
    n = 2 * r

    # Create a list of vertices for the rotating circle: these are 1,..., n-1.
    circle = list(range(0, n - 1))
    fixed = n - 1  # fixed vertex
    
    rounds = []
    for _ in range(n - 1):  # n-1 rounds = 2r-1 rounds
        current_round = []

        # Pair the fixed vertex with the first vertex in the circle.
        current_round.append((circle[0], fixed))

        # Pair the remaining vertices in mirror order.
        for j in range(1, r):
            current_round.append((circle[j], circle[-j]))
        rounds.append(current_round)

        # Rotate the circle by moving the last element to the front.
        circle = [circle[-1]] + circle[:-1]

    return rounds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--nprocs', type=int)
    args = parser.parse_args()

    # Generate off-diagonal strata
    rounds = factorize_pairs(args.nprocs)

    # Correct points
    rounds_ = []
    for round in rounds: 
        round_ = []
        for i, j in round: 
            if i > j: 
                round_.append((i,j))
            else: 
                round_.append((j,i))
        rounds_.append(round_)

    # Validate
    validate_rounds(args.nprocs, rounds_)

    # Add diagonals
    d1_stratum = [(r, r) for r in range(args.nprocs)]
    d2_stratum = [(r, r) for r in range(args.nprocs, 2 * args.nprocs)]
    rounds_ = [d1_stratum, d2_stratum] + rounds_

    # Construct tensor
    n_rows = args.nprocs * (2 * args.nprocs + 1)
    strata = torch.zeros(n_rows, 3, dtype=torch.int32)
    idx = 0
    for s, round in enumerate(rounds_):
        for i, j in round: 
            strata[idx] = torch.tensor([s, i, j])
            idx += 1

    # Write
    path = os.path.join('strata', f'nprocs-{args.nprocs}.pt')
    torch.save(strata, path)
