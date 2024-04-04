import argparse
import os
import torch

from factor_analyzer.rotator import Rotator



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-3)
    args = parser.parse_args()

    # Rotate models estiamted from training and full datasets
    rotator = Rotator(method='varimax', max_iter=args.max_iter, tol=args.tol)
    files_in = ['model-full.pth', 'model-train.pth']
    files_out = ['model-full-rot.pth', 'model-train-rot.pth']
    for f_in, f_out in zip(files_in, files_out): 

        path_in = os.path.join(args.dir_out, f_in)
        path_out = os.path.join(args.dir_out, f_out)

        model = torch.load(path_in)
        loads = model['loads'].numpy()
        loads_rot = rotator.fit_transform(loads)
        model['loads'] = torch.from_numpy(loads_rot)
        torch.save(model, path_out)
