import argparse
import numpy as np
import os
import scipy
import torch
from petsc4py import PETSc
from slepc4py import SLEPc
from typing import Generator, Tuple

from config import load_config
from utils import (
    compute_covariance,
    gen_points,
    gen_tensors,
    multiply_list
)



def get_thresholded_epairs(err_cov, cutoff):
    """Implictly project the passed error covariance onto the space of 
    symmetric matrices having eigenvalues greater than some threshold by 
    computing the eigenpairs of said projection."""

    # Create PETSc matrix from SciPy CSR matrix
    err_cov_petsc = PETSc.Mat().createAIJ(
        size=err_cov.shape, 
        csr=(err_cov.indptr, err_cov.indices, err_cov.data)
    )

    # Define the eigenproblem solver
    solver = SLEPc.EPS().create()
    solver.setOperators(err_cov_petsc)
    solver.setProblemType(SLEPc.EPS.ProblemType.HEP)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)

    # Set the tolerance and maximum iterations (per solve call)
    # TODO: Pass these as parameters to this function?
    solver.setTolerances(1e-8, 1000)

    k = 10
    max_iter = 10  # maximum number of solve calls
    for _ in range(max_iter):

        # Set the number of eigenpairs to compute then solve
        solver.setDimensions(k)
        solver.solve()

        nconv = solver.getConverged()
        evals = []
        evecs = []
        for j in range(nconv):

            eval = solver.getEigenpair(j)
            if eval.imag != 0:  # TODO: Remove this check?
                print(f"e-val {j} has imaginary part: {eval.real} + {eval.imag}i")
            if eval.real > cutoff:
                evals.append(eval.real)
                vr, vi = err_cov_petsc.createVecs()
                solver.getEigenvector(j, vr, vi)
                evecs.append(vr.getArray())

        evals = np.array(evals)
        evecs = np.array(evecs).T

        if len(evals) < k:
            break
        else:
            k *= 2

    return evals, evecs


def estimate_err_cov(
        init_err_cov: np.ndarray, 
        cutoff: int = 0, 
        tol: float = 1e-8, 
        max_iters: int = 10000
    ) -> Tuple[np.ndarray]:
    """Obtain a low-rank estimate of the banded error covariance via 
    projection onto convex sets (a.k.a., POCS).

    Args:
        init_err_cov (np.ndarray): Initial error covariance (sparse csr matrix).
        cutoff (int, optional): Eigenvalue cutoff. Defaults to 0.02.
        tol (float, optional): Tolerance used for stop criterion. Defaults to 1e-8.
        max_iters (int, optional): Maximum number of iterations. Defaults to 10000.

    Returns:
        Tuple[np.ndarray]: Eigenvalues and eigenvectors of estimate.
    """

    row_idx, col_idx = init_err_cov.nonzero()
    n_vars = init_err_cov.shape[0]
    b_new = init_err_cov
    b_old = None
    converged = False
    for i in range(max_iters):

        if i > 0: 

            # Project onto space of banded matrices
            data = np.sum(evals * evecs[row_idx] * evecs[col_idx], axis=1)
            b_new = scipy.sparse.csr_matrix(
                (data, (row_idx, col_idx)), 
                shape=(n_vars, n_vars)
            )

            # Compute difference and break if converged
            diff = scipy.sparse.linalg.norm(b_new - b_old) / n_vars ** 2
            print(f"iter = {i + 1} | err = {diff}")
            if diff <= tol:
                converged = True
                break
            
        b_old = b_new

        # Implicitly project onto space of low-rank symmetric matrices with 
        # eigenvalues greater than cutoff
        evals, evecs = get_thresholded_epairs(b_new, cutoff)
    
    if not converged:
        print(f"No convergence after {max_iters} iterations.")
    return evals, evecs  # used to compute low rank symm psd approx to estimated error covariance


def invert_err_cov_old(evals, evecs, phi):
    n = evecs.shape[0]
    rk = evecs.shape[1]
    u_mat = evecs @ np.diag(np.sqrt(evals))
    inv_d_mat = np.eye(n) / phi
    inv_est_err_cov = (
        inv_d_mat - 
        inv_d_mat @ u_mat @ np.linalg.inv(np.eye(rk) + u_mat.T @ inv_d_mat @ u_mat) @ u_mat.T @ inv_d_mat
    )
    return inv_est_err_cov


def invert_err_cov(
        evals: np.ndarray, 
        evecs: np.ndarray, 
        phi: float, 
        batch_size: int
    ) -> Generator:
    """Inverts UUt + D where U = evecs @ diag(sqrt(evals)) and D = phi*I
    via a specialized Woodbury identity:  
        (UUt + D) = D_inv - D_inv @ U @ A_inv @ Ut @ D_inv
    where A = I - Ut @ D_inv @ U. Yields solution in batches.
    """
    
    n_vars = evecs.shape[0]
    rk = evecs.shape[1]
    u_mat = evecs @ np.diag(np.sqrt(evals))

    # Compute inverse of A in memory-efficient manner
    a_mat = np.eye(rk, dtype=np.float64) +  (u_mat.T @ u_mat) / phi
    a_mat_inv = scipy.linalg.inv(a_mat)

    # Yield inverse in row-wise batches
    start = 0
    while start < n_vars:
        sz = min(batch_size, n_vars - start)
        d_inv_mat_ = np.hstack((  # sz rows of d_inv_mat
            np.zeros((sz, start), dtype=np.float64),
            np.eye(sz, dtype=np.float64) / phi,
            np.zeros((sz, n_vars - start - sz), dtype=np.float64)
        ))
        u_mat_ = u_mat[start:(start + sz)]  # sz rows of u_mat
        yield d_inv_mat_ - (u_mat_ @ a_mat_inv @ u_mat.T) / (phi ** 2)
        start += sz



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--dir_out_scratch', type=str)
    parser.add_argument('--regime', type=int, choices=[1, 2, 3])
    parser.add_argument('--split', type=str)
    parser.add_argument('--sz_space', nargs='+', type=int)
    parser.add_argument('--est_method_loads', type=str, choices=['lbfgs', 'dsgd', 'dssgd'])
    parser.add_argument('--delta_true', type=float)
    parser.add_argument('--eval_cutoff', type=float)
    parser.add_argument('--phi', type=float)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    n_vars = multiply_list(args.sz_space)

    # Validate arguments
    if args.regime == 1:
        assert args.delta_true is not None, "Must pass delta_true for Regime 1!"
    if args.regime == 2: 
        assert args.dir_out_scratch is not None, "Must pass dir_out_scratch for Regime 2!"
    if args.regime == 3: 
        assert args.dir_out_scratch is not None, "Must pass dir_out_scratch for Regime 2!"


    if args.regime == 1: 

        # Read in error function set
        path = os.path.join(args.dir_out, 'errs.pt')
        err = torch.load(path)  # (J, T, S)
        err = torch.transpose(torch.transpose(err, 0, 2), 1, 2)  # (S, J, T)
        n_time = err.size(2)

        # Estimate error covariance from error function set
        points_loader = gen_points(
            args.sz_space, args.delta_true, n_vars, 
            off_band=False,
            exclude_upp_tri=False,
            as_numpy=False
        )
        row_idx_list = []
        col_idx_list = []
        data_list = []
        for points in points_loader:
            n_points = len(points)
            vals = torch.zeros(n_points, dtype=torch.float64)
            for i in range(n_points):
                vals[i] = (err[points[i,0]] * err[points[i,1]]).sum() / n_time
            row_idx_list.append(points[:,0].numpy())
            col_idx_list.append(points[:,1].numpy())
            data_list.append(vals.numpy())
        data = np.concatenate(data_list)
        row_idx = np.concatenate(row_idx_list)
        col_idx = np.concatenate(col_idx_list)
        err_cov = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_vars, n_vars)
        )

        # Obtain low-rank approximation for faster inversion
        evals, evecs = get_thresholded_epairs(err_cov, args.eval_cutoff)

        # Invert low-rank error covariance via Woodbury
        inv_err_cov_loader = invert_err_cov(
            evals, evecs, 
            phi=args.phi, 
            batch_size=args.batch_size
        )
        for i, rows in enumerate(inv_err_cov_loader):
            path = os.path.join(args.dir_out, f'inv-err-cov_r{args.regime}_{i}.npy')
            np.save(path, rows)

    if args.regime in [2, 3]:

        # Read in the appropriate loadings
        if args.regime == 2: 
            path = os.path.join(args.dir_out, 'loads.pt')
            loads = torch.load(path).t().numpy()
        if args.regime == 3: 
            path = os.path.join(
                args.dir_out, 
                f'model-{args.est_method_loads}-{args.split}.pth'
            )
            loads = torch.load(path)['loads'].data.numpy()

        # Prepare initial error covariance, a sparse csr tensor equal to 
        #       band(C_hat - LL^T)
        row_idx_list = []
        col_idx_list = []
        data_list = []
        dir_cov = os.path.join(args.dir_out_scratch, 'cov')
        points_loader = gen_tensors(dir_cov, 'points')
        cov_loader = gen_tensors(dir_cov, 'cov')
        for points, cov in zip(points_loader, cov_loader): 
            row_idx_list.append(points[:,0])
            col_idx_list.append(points[:,1])
            glob = (loads[points[:,0]] * loads[points[:,1]]).sum(axis=1)
            data_list.append(cov - glob)
        data = np.concatenate(data_list)
        row_idx = np.concatenate(row_idx_list)
        col_idx = np.concatenate(col_idx_list)
        init_err_cov = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_vars, n_vars)
        )

        # Estimate low-rank error covariance    
        evals, evecs = estimate_err_cov(init_err_cov, cutoff=args.eval_cutoff)

        # Invert low-rank error covariance via Woodbury
        inv_err_cov_loader = invert_err_cov(
            evals, evecs, 
            phi=args.phi, 
            batch_size=args.batch_size
        )

        # Write inverse in batches
        for i, rows in enumerate(inv_err_cov_loader):
            path = os.path.join(args.dir_out, 'err-cov', f'inv-err-cov_r{args.regime}_{i}.npy')
            np.save(path, rows)

