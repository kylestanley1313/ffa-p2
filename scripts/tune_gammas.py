import argparse
import os
import sys
import torch
from functools import partial

from utils import (
    execute_script,
    gen_tensors,
    load_config,
)


def compute_loss(idx, dataloader, loads, facs):
    """Computes least squares loss:
            1/(MT) trace((X - LF)^T (X - LF))
    """
    n_idx = len(idx)
    n_time = facs.shape[1]

    out = 0
    start = 0
    for data in dataloader:
        sz = len(data)
        idx_mask = torch.logical_and(idx >= start, idx < start + sz)
        idx_ = idx[idx_mask]
        errs = data[idx_ - start] - loads.T[idx_] @ facs
        out += torch.sum(errs ** 2)
        start += sz
    
    return out / n_idx / n_time



def compute_cv_loss_for_gammas(
        gammas, 
        sub_nums,
        n_folds,
        config,
        dir_out,
        dir_out_scratch,
        est_method_loads,
        rot_method,
        regime,
        idx_valid_folds,
        loads,
        batch_size,
        path_mask=None
    ):

    # Path preparation
    path_facs = lambda v: os.path.join(
        dir_out,
        f'facs_split-train_v-{v}_m-rbels_rot-{rot_method}_reg-{regime}.pt'
    )
    path_gammas = os.path.join(dir_out, f'tmp_gammas_rot-{rot_method}_reg-{regime}.pt')    
    if gammas.ndim == 1: 
        gammas = gammas.unsqueeze(0)
    torch.save(gammas, path_gammas)

    # Build Execution parameters
    path_script = os.path.join(config.root, 'scripts', f'estimate_factor_scores.py')
    flags = {
        'config': args.config,
        'dir_out': dir_out,
        'dir_out_scratch': dir_out_scratch,
        'est_method': 'rbels',
        'sub_nums': sub_nums,
        'path_gammas': path_gammas,
        'split': 'train',
        'est_method_loads': est_method_loads,
        'rot_method': rot_method,
        'regime': regime,
        'batch_size': batch_size,
        'agg_subs': None
    }
    if path_mask is not None:
        flags['path_mask'] = path_mask

    loss = 0
    for fold in range(n_folds):

        # Estimate factor scores
        flags['fold'] = fold
        code = execute_script(path_script, flags, False)
        if code != 0: 
            sys.exit(code)
        facs = torch.load(path_facs(fold))

        for i, n in enumerate(sub_nums): 

            # Create dataloader for validation data
            dataloader = gen_tensors(
                dir=os.path.join(dir_out_scratch, 'data'),
                prefix=f'data-space_split-full_n-{n}_',
                batch_size=args.batch_size, 
                sort_by=('i', int)
            )

            # Compute validation loss
            loss += compute_loss(
                idx_valid_folds[fold], 
                dataloader, 
                loads, facs[i],
            ) / len(sub_nums)

        os.remove(path_facs(fold))
    os.remove(path_gammas)

    return loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument(
        '--sub_nums', type=int, nargs='+',
        help=("The subjects used for tuning. When sharing across subjects, " 
              "the length of this list may be less than `n_sub_in_rep`.")
    )
    parser.add_argument(
        '--gamma_grid', type=float, nargs='+',
        default = [
            1e-6, 
            1e-5, 
            1e-4, 
            1e-3, 
            1e-2, 
            1e-1, 
        ]
    )
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--rot_method', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2])
    parser.add_argument('--path_mask', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--radius', type=int, default=2)
    # parser.add_argument('--universal_param', action='store_true')
    parser.add_argument('--share_across_subs', action='store_true')
    parser.add_argument('--share_across_k', action='store_true')
    parser.add_argument(
        '--n_sub_in_rep', type=int,
        help=("The number of subjects in the repetition. This may be greater"
              "than the length of sub_nums when sharing across subjects.")
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Directory/path preparation
    dir_data = os.path.join(args.dir_out_scratch, 'data')
    
    # Get loadings
    if args.regime == 1: 
        path = os.path.join(args.dir_out, 'loads.pt')
        loads = torch.load(path)
        if args.path_mask is not None: 
            mask = torch.flatten(torch.load(args.path_mask))
            loads = loads[:,mask]
    else: 
        path = os.path.join(args.dir_out, f'model-{args.est_method_loads}-full-{args.rot_method}-smooth-shrink.pth')
        loads = torch.load(path)['loads'].t()

    # Get validation indices for each fold
    idx_valid_folds = []
    for v in range(args.n_folds):
        path = os.path.join(dir_data, f'idx-space_split-valid_v-{v}_.pt')
        idx_valid_folds.append(torch.load(path))

    # Partially define CV loss function
    compute_cv_loss_for_gammas = partial(
        compute_cv_loss_for_gammas,
        n_folds=args.n_folds,
        config=config,
        dir_out=args.dir_out,
        dir_out_scratch=args.dir_out_scratch,
        # sub_nums=args.sub_nums,
        est_method_loads=args.est_method_loads,
        rot_method=args.rot_method,
        regime=args.regime,
        idx_valid_folds=idx_valid_folds,
        loads=loads,
        batch_size=args.batch_size,
        path_mask=args.path_mask
    )

    # Set other globals
    n_facs = len(loads)
    max_idx = len(args.gamma_grid) - 1

    if args.share_across_subs: 

        # Initialize loss
        best_idx = [0] * n_facs
        gammas = torch.tensor([args.gamma_grid[_] for _ in best_idx])
        gammas = gammas.unsqueeze(0).repeat(len(args.sub_nums), 1)
        best_loss = compute_cv_loss_for_gammas(gammas, args.sub_nums)
        print(f"gammas = {gammas[0].tolist()} | valid_loss = {best_loss}", flush=True)
        
        if args.share_across_k:  # universal descent
            
            # Set index deltas
            deltas = torch.arange(1, args.radius + 1)

            for i in range(1, args.max_iters + 1):

                best_loss_prev = best_loss

                # Create candidate indices
                indices = deltas + best_idx[0]
                indices = indices[indices <= max_idx].tolist()
                if len(indices) == 0:  # no more candidate gammas
                    break

                for idx in indices: 

                    # Compute validation loss
                    test_idx = torch.tensor([idx] * n_facs, dtype=torch.int32)
                    gammas = torch.tensor([args.gamma_grid[_] for _ in test_idx])
                    gammas = gammas.unsqueeze(0).repeat(len(args.sub_nums), 1)
                    valid_loss = compute_cv_loss_for_gammas(gammas, args.sub_nums)
                    print(f"gammas = {gammas[0].tolist()} | valid_loss = {valid_loss}", flush=True)

                    # Update bests
                    if valid_loss < best_loss:
                        best_idx = [idx] * n_facs
                        best_loss = valid_loss

                print(f"iter = {i} | loss = {best_loss}", flush=True)
                if best_loss_prev - best_loss < args.tol: 
                    break
                best_loss_prev = best_loss

        else:  # coordinate descent

            # Set index deltas
            tmp1 = -torch.flip(torch.arange(1, args.radius + 1), dims=[0])
            tmp2 = torch.arange(1, args.radius + 1)
            deltas = torch.concatenate((tmp1, tmp2))  

            # Tune gammas via coordinate descent
            for i in range(1, args.max_iters + 1):

                best_loss_prev = best_loss

                for k in range(n_facs):

                    best_idx_k = best_idx[k]
                    best_loss_k = best_loss

                    # Create candidate indices
                    indices = deltas + best_idx[k]
                    mask = torch.logical_and(indices >= 0, indices <= max_idx)
                    indices = indices[mask].tolist()
                    if len(indices) == 0:  # no more candidate gammas
                        break

                    for idx in indices: 

                        # Set gamma
                        test_idx = best_idx.copy()
                        test_idx[k] = idx
                        gammas = torch.tensor([args.gamma_grid[_] for _ in test_idx])
                        gammas = gammas.unsqueeze(0).repeat(len(args.sub_nums), 1)
                        valid_loss = compute_cv_loss_for_gammas(gammas, args.sub_nums)
                        print(f"gammas = {gammas[0].tolist()} | valid_loss = {valid_loss}", flush=True)

                        # Update bests
                        if valid_loss < best_loss_k:
                            best_idx_k = idx
                            best_loss_k = valid_loss

                    # Update bests
                    best_idx[k] = best_idx_k
                    best_loss = best_loss_k

                print(f"iter = {i} | loss = {best_loss}", flush=True)
                if best_loss_prev - best_loss < args.tol: 
                    break
                best_loss_prev = best_loss
        
        # Save best gammas
        fname = f'gammas_rot-{args.rot_method}_reg-{args.regime}.pt'
        path = os.path.join(args.dir_out, fname)
        best_gammas = torch.tensor([args.gamma_grid[_] for _ in best_idx])
        best_gammas = best_gammas.unsqueeze(0).repeat(args.n_sub_in_rep, 1)
        torch.save(best_gammas, path)


    else:  # subject-wise tuning

        for n in args.sub_nums: 
            print(f"---------- n = {n} ----------")

            # Initialize loss
            best_idx = [0] * n_facs
            gammas = torch.tensor([args.gamma_grid[_] for _ in best_idx])
            best_loss = compute_cv_loss_for_gammas(gammas, [n])
            print(f"gammas = {gammas.tolist()} | valid_loss = {best_loss}", flush=True)

            if args.share_across_k:  # universal descent

                # Set index deltas
                deltas = torch.arange(1, args.radius + 1)

                for i in range(1, args.max_iters + 1):

                    best_loss_prev = best_loss

                    # Create candidate indices
                    indices = deltas + best_idx[0]
                    indices = indices[indices <= max_idx].tolist()
                    if len(indices) == 0:  # no more candidate gammas
                        break

                    for idx in indices: 

                        # Compute validation loss
                        test_idx = torch.tensor([idx] * n_facs, dtype=torch.int32)
                        gammas = torch.tensor([args.gamma_grid[_] for _ in test_idx])
                        # print(f"gammas (tune) = {gammas}")
                        valid_loss = compute_cv_loss_for_gammas(gammas, [n])
                        print(f"gammas = {gammas.tolist()} | valid_loss = {valid_loss}", flush=True)

                        # Update bests
                        if valid_loss < best_loss:
                            best_idx = [idx] * n_facs
                            best_loss = valid_loss

                    print(f"iter = {i} | loss = {best_loss}", flush=True)
                    if best_loss_prev - best_loss < args.tol: 
                        break
                    best_loss_prev = best_loss


            else:  # coordinate descent
                
                # Set index deltas
                tmp1 = -torch.flip(torch.arange(1, args.radius + 1), dims=[0])
                tmp2 = torch.arange(1, args.radius + 1)
                deltas = torch.concatenate((tmp1, tmp2))  

                # Tune gammas via coordinate descent
                for i in range(1, args.max_iters + 1):

                    best_loss_prev = best_loss

                    for k in range(n_facs):

                        best_idx_k = best_idx[k]
                        best_loss_k = best_loss

                        # Create candidate indices
                        indices = deltas + best_idx[k]
                        mask = torch.logical_and(indices >= 0, indices <= max_idx)
                        indices = indices[mask].tolist()
                        if len(indices) == 0:  # no more candidate gammas
                            break

                        for idx in indices: 

                            # Set gamma
                            test_idx = best_idx.copy()
                            test_idx[k] = idx
                            gammas = torch.tensor([args.gamma_grid[_] for _ in test_idx])
                            valid_loss = compute_cv_loss_for_gammas(gammas, [n])
                            print(f"gammas = {gammas.tolist()} | valid_loss = {valid_loss}", flush=True)

                            # Update bests
                            if valid_loss < best_loss_k:
                                best_idx_k = idx
                                best_loss_k = valid_loss

                        # Update bests
                        best_idx[k] = best_idx_k
                        best_loss = best_loss_k

                    print(f"iter = {i} | loss = {best_loss}", flush=True)
                    if best_loss_prev - best_loss < args.tol: 
                        break
                    best_loss_prev = best_loss

            # Save best gammas
            fname = f'n-{n}_rot-{args.rot_method}_reg-{args.regime}.pt'
            path = os.path.join(args.dir_out, 'gammas', fname)
            best_gammas = torch.tensor([args.gamma_grid[_] for _ in best_idx])
            torch.save(best_gammas, path)

        # Aggregate gammas
        gammas = []
        for n in args.sub_nums: 
            fname = f'n-{n}_rot-{args.rot_method}_reg-{args.regime}.pt'
            path = os.path.join(args.dir_out, 'gammas', fname)
            gammas.append(torch.load(path))
            os.remove(path)
        gammas = torch.stack(gammas, dim=0)
        fname = f'gammas_rot-{args.rot_method}_reg-{args.regime}.pt'
        path = os.path.join(args.dir_out, fname)
        torch.save(gammas, path)


