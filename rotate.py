import argparse
import numpy as np
import os
import torch

from config import load_config



# NOTE: Copied from https://github.com/EducationalTestingService/factor_analyzer/blob/main/factor_analyzer/rotator.py
#       Installation of factor_analyzer was taking too long
def varimax(loadings, max_iter, tol, normalize=False):
        """
        Perform varimax (orthogonal) rotation, with optional Kaiser normalization.
        """
        X = loadings.copy()
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if normalize:
            normalized_mtx = np.apply_along_axis(
                lambda x: np.sqrt(np.sum(x**2)), 1, X.copy()
            )
            X = (X.T / normalized_mtx).T

        # initialize the rotation matrix
        # to N x N identity matrix
        rotation_mtx = np.eye(n_cols)

        d = 0
        for _ in range(max_iter):
            old_d = d

            # take inner product of loading matrix
            # and rotation matrix
            basis = np.dot(X, rotation_mtx)

            # transform data for singular value decomposition using updated formula :
            # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
            diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
            transformed = X.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

            # perform SVD on
            # the transformed matrix
            U, S, V = np.linalg.svd(transformed)

            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U, V)
            d = np.sum(S)

            # check convergence
            if d < old_d * (1 + tol):
                break

        # take inner product of loading matrix
        # and rotation matrix
        X = np.dot(X, rotation_mtx)

        # de-normalize the data
        if normalize:
            X = X.T * normalized_mtx
        else:
            X = X.T

        # convert loadings matrix to data frame
        loadings = X.T.copy()
        return loadings, rotation_mtx



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--est_method_loads', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-3)
    args = parser.parse_args()

    config = load_config(args.config)

    # Set input/output paths
    f_in = f'model-{args.est_method_loads}-{args.split}.pth'
    f_out = f'model-{args.est_method_loads}-{args.split}-rot.pth'
    path_in = os.path.join(args.dir_out, f_in)
    path_out = os.path.join(args.dir_out, f_out)

    # Rotate
    model = torch.load(path_in)
    loads = model['loads'].numpy()
    loads_rot, _ = varimax(loads, args.max_iter, args.tol)
    model['loads'] = torch.from_numpy(loads_rot)
    torch.save(model, path_out)

