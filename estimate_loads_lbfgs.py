import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmarking import size_dist_obj, time_dist_fcn
from config import load_config
from utils.data import CentralizedCovarianceDataset
from utils.model import LowRankCovariance
from utils.utils import (
    create_second_difference_matrix,
    gen_seeds, 
    init_process,
    loss_fcn,
    multiply_list,
    penalty_fcn,
    remove_file,
)


def process_epoch(model, dataset, loss_fcn, penalty_fcn, optimizer):

    def closure():
        pred = model(dataset.points)
        loss = loss_fcn(pred, dataset.cov)
        penalty = penalty_fcn(model.loads)
        objective = loss + penalty
        optimizer.zero_grad()
        objective.backward()
        return objective
    
    optimizer.step(closure)


def compute_objective(
        model, 
        dataset, 
        loss_fcn,
        penalty_fcn  
    ):
    loss = loss_fcn(model(dataset.points), dataset.cov)
    penalty = penalty_fcn(model.loads)
    return loss, penalty


def train(
        dir_out, 
        dir_out_scratch, 
        split, 
        sz_space, 
        n_facs, 
        alpha, 
        lr, 
        history_size, 
        tol, 
        patience, 
        max_epochs
    ): 

    # Set directories and paths
    dir_cov = os.path.join(dir_out_scratch, 'cov-lbfgs')
    path_model = os.path.join(dir_out, f'model-lbfgs-{split}.pth')

    dataset = CentralizedCovarianceDataset(dir_cov, split)
    n_vars = multiply_list(sz_space)
    path_init = os.path.join(dir_out, f'init-loads-{split}.pt')
    model = LowRankCovariance(n_vars, n_facs, path_init)

    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=lr, 
        history_size=history_size
    )
    loss_fcn_ = partial(loss_fcn, n_vars=n_vars)
    diff_mat = create_second_difference_matrix([n_vars])
    penalty_fcn_ = partial(penalty_fcn, alpha=alpha, diff_mat=diff_mat)

    last_objective = float('inf')
    epochs_waited = 0
    early_stop = torch.tensor(False)
    diverged = torch.tensor(False)
    for epoch in range(max_epochs):
        process_epoch(
            model, dataset, 
            loss_fcn_, penalty_fcn_, 
            optimizer
        )
        loss, penalty = compute_objective(
            model, dataset, 
            loss_fcn_, penalty_fcn_
        )
        objective = loss + penalty
        print(f"epoch = {epoch + 1} | objective = {objective.item()}")

        # Handle divergence
        if torch.isnan(objective) or torch.isinf(objective):
            diverged = torch.tensor(True)

        else:
            # Handle early stopping
            if abs(objective - last_objective) > tol:
                epochs_waited = 0
            else: 
                epochs_waited += 1
                if epochs_waited >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    early_stop = torch.tensor(True)
            last_objective = objective

        if diverged or early_stop:
            break


    if diverged: 
        print(f"Error: Divergence after {epoch + 1} epochs.")

    else: 
        if not early_stop:
            print(f"Warning: No convergence after {epoch + 1} epochs.")
        
        # Save model (even if no convergence)
        torch.save(model.state_dict(), path_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--split', type=str, choices=['full', 'train', 'valid'])
    parser.add_argument('--sz_space', type=int, nargs='+')
    parser.add_argument('--n_facs', type=int)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--history_size', type=int)
    parser.add_argument('--tol', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--max_epochs', type=int, default=100)
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set directories and paths
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    dir_cov = os.path.join(args.dir_out_scratch, 'cov-dist')
    dir_bench = os.path.join(args.dir_out, 'bench')
    path_init = os.path.join(args.dir_out, f'init-loads-{args.split}.pt')
    path_model = os.path.join(args.dir_out, f'model-lbfgs-{args.split}.pth')
    other_bench_path = os.path.join(dir_bench, 'other-lbfgs.csv')


    # ---------- ESTIMATION ---------- #
    print("Fitting model...")

    train(
        dir_out=args.dir_out,
        dir_out_scratch=args.dir_out_scratch,
        split=args.split,
        sz_space=args.sz_space,
        n_facs=args.n_facs,
        alpha=args.alpha,
        lr=args.lr,
        history_size=args.history_size,
        tol=args.tol,
        patience=args.patience,
        max_epochs=args.max_epochs
    )



    # suffix = args.dir_out.split('out/')[-1].replace('/', '_')
    # path_shared = os.path.join(config.dir_shared, f'shared_{suffix}')
    # remove_file(path_shared)
    # remove_file(path_model)

    # mp.set_start_method('spawn')
    # processes = []
    # for rank in range(args.world_size):
    #     p = mp.Process(
    #         target=init_process, 
    #         args=(rank, args.world_size, train, path_shared, 'gloo'),
    #         kwargs={
    #             'dir_out': args.dir_out,
    #             'dir_out_scratch': args.dir_out_scratch,
    #             'split': args.split,
    #             'sz_space': args.sz_space,
    #             'n_facs': args.n_facs,
    #             'alpha': args.alpha,
    #             'lr': args.lr,
    #             'history_size': args.history_size,
    #             'tol': args.tol,
    #             'patience': args.patience,
    #             'max_epochs': args.max_epochs,
    #             'benchmark': args.benchmark,
    #         }
    #     )
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
