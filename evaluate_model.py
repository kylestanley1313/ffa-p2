import argparse
import os
import torch

from benchmarking import aggregate_benchmarks
from utils import refresh_directory, write_rows_to_csv
from utils_plotting import (
    plot_line_for_1d_loads,
    plot_heatmap_for_2d_loads,
    plot_heatmap_for_3d_loads
)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--grid_shape', type=int, nargs='+')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    # Globals
    dir_out = os.path.join('out', args.dir_out)
    dir_bench = os.path.join(dir_out, 'bench')

    # Prepare the results directory
    dir_res = os.path.join(dir_out, 'results')
    refresh_directory(dir_res)

    if args.plot:
        
        path_model = os.path.join(dir_out, 'cov-model.pth')
        loads = torch.load(path_model)['loads'].data
        num_facs = loads.shape[-1]
        loads = loads.reshape(args.grid_shape + [num_facs])
        ndim = len(args.grid_shape)
        dims = list(range(ndim + 1))
        loads = loads.permute(dims[-1:] + dims[:ndim])
        for k in range(num_facs):
            if ndim == 1:
                path = os.path.join(dir_res, f'loads_k-{k}.png')
                plot_line_for_1d_loads(loads.data, k, path=path)
            elif ndim == 2:
                path = os.path.join(dir_res, f'loads_k-{k}.png')
                plot_heatmap_for_2d_loads(loads.data, k, path=path)
            elif ndim == 3:
                for p in [0.25, 0.5, 0.75]:
                    z = int(p * args.grid_shape[2])
                    path = os.path.join(dir_res, f'loads_k-{k}_z-{z}.png')
                    plot_heatmap_for_3d_loads(loads.data, k, z, path=path) 
            else:
                print(f"`grid_shape` must have length <= 3!")

    if args.benchmark:

        path_bench = os.path.join(dir_res, 'bench.csv')
        rows = []
        rows += aggregate_benchmarks(dir_bench, 'process_epoch', 'mean')
        rows += aggregate_benchmarks(dir_bench, 'dataset', 'mean')
        rows += aggregate_benchmarks(dir_bench, 'other')
        write_rows_to_csv(path_bench, rows)



