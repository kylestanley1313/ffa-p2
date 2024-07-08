import argparse
import os
import pandas as pd
import torch

from benchmarking import aggregate_benchmarks
from config import load_config
from utils import (
    l2_norm,
    load_yaml,
    multiply_list,
    refresh_directory, 
    write_rows_to_csv,
)
from utils.plotting import (
    plot_line_for_1d_loads,
    plot_heatmap_for_2d_loads,
    plot_heatmap_for_3d_loads,
    plot_side_by_side_heatmaps
)

# NOTE: (Script Outline)
#   - Required metrics (computed in this script)
#       * LE1: ||G - G_hat|| / ||G| ; from L and L_hat
#       * LE2: train_err and time per epoch; from /bench 
#       * FE1: ||F - F_hat|| / F; from F and F_hat
#       * FE2: Same
#       * FE3: Same

# dir_out_sim = '/Users/kylestanley/repos/ffa-p2-priv/out/test-1/sim-0'
# path_true = os.path.join(dir_out_sim, 'loads.pt')
# path_init = os.path.join(dir_out_sim, 'rep-0', 'init-loads-full.pt')
# path_lbfgs = os.path.join(dir_out_sim, 'rep-0', 'model-lbfgs-full.pth')
# loads_true = torch.load(path_true)
# loads_init = torch.load(path_init)
# loads_lbfgs = torch.load(path_lbfgs)['loads'].data

# plot_line_for_1d_loads(loads_true, 0)
# plot_line_for_1d_loads(loads_init.t(), 0)
# plot_line_for_1d_loads(loads_lbfgs.t(), 0)

# exit(0)



def assign_load_scheme(row):
    if row['load_variety'].startswith('Bump'):
        prefix = 'BL'
    elif row['load_variety'].startswith('Net'):
        prefix = 'NET'
    else: 
        raise Exception("Invalid loading variety!")
    return prefix + str(row['n_facs'])


def assign_err_scheme(row):

    if row['err_variety'].startswith('GaussProc_Bump'):
        prefix = 'BE'
    elif row['load_variety'].startswith('GaussProc_BSpline'):
        prefix = 'BS'
    else: 
        raise Exception("Invalid loading variety!")
    
    if row['delta'] == 0.05:
        suffix = '05'
    elif row['delta'] == 0.1:
        suffix = '10'
    else: 
        raise Exception("Invalid delta!")

    return prefix + suffix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    des = load_yaml(
        os.path.join(config.root, 'designs', f'{args.design}.yml')
    )
    dir_design = os.path.join(config.root, 'designs', args.design)
    dir_out = os.path.join(config.root, 'out', args.design)
    sim_ids = os.listdir(dir_out)

    # Build dataframe of simulation results
    columns = [
        'sim_id', 'rep_id', 'est_method',
        'load_variety', 'n_facs', 'err_variety', 'delta',  # figure columns
        'prop_glob', 'fac_kern_length', 'n_time',  # figure rows
        'err'
    ]
    df = pd.DataFrame(columns=columns)
    for sim_id in sim_ids: 

        # Read true loadings
        dir_sim = os.path.join(config.root, 'designs', args.design, sim_id)
        dir_out_sim = os.path.join(config.root, 'out', args.design, sim_id)
        loads = torch.load(os.path.join(dir_out_sim, 'loads.pt'))
        
        # Reshape true loads
        n_space = multiply_list(loads.shape[1:])
        loads = loads.reshape(loads.size(0), n_space)
        g = loads.t() @ loads

        for r in range(des['n_reps']):
            
            # Load repetition
            rep = load_yaml(os.path.join(dir_sim, f'rep-{r}.yml'))

            # Compute estimation error for each method then add row to dataframe
            dir_out_rep = os.path.join(dir_out_sim, f'rep-{r}')
            for method in ['lbfgs', 'dsgd', 'dssgd']:

                path = os.path.join(dir_out_rep, f'model-{method}-full.pth')
                est_loads = torch.load(path)['loads'].data
                g_hat = est_loads @ est_loads.t()
                err = torch.norm(g_hat - g) / torch.norm(g)

                row = {
                    'sim_id': sim_id,
                    'rep_id': f'rep-{r}',
                    'est_method': method,
                    'load_variety': rep['load_scheme'],
                    'n_facs': rep['n_facs'],
                    'err_variety': rep['err_scheme'],
                    'delta': rep['delta'],
                    'prop_glob': rep['prop_global'],
                    'fac_kern_length': rep['factor_kernel_length'],
                    'n_time': rep['n_time'],
                    'err': err.item()
                }
                df = pd.concat([df, pd.DataFrame([row])])

    # Assign schemes
    df['load_scheme'] = df.apply(assign_load_scheme, axis=1)
    df['err_scheme'] = df.apply(assign_err_scheme, axis=1)

    # Assign errors relative to lbfgs
    df['err_lbfgs'] = pd.merge(
        df, df[df.est_method == 'lbfgs'], 
        how='left',
        left_on=['sim_id', 'rep_id'],
        right_on=['sim_id', 'rep_id']
    )['err_y'].to_list()
    df['rel_err'] = df['err'] / df['err_lbfgs']

    for i in range(0, 96, 10):
        sz = min(10, 96-i)
        print(df[i:(i+sz)])


    exit(0)

    df_agg = df.groupby([
        'est_method', 
        'load_scheme', 'err_scheme', 
        'prop_glob', 'fac_kern_length', 'n_time']).agg(
            mean_err=('err', 'mean'),
            std_err=('err', 'std')
        ).reset_index()
    print(df_agg)

        


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--est_methods', nargs='+', type=str, 
#         default=['lbfgs', 'dsgd', 'dssgd']
#     )
#     parser.add_argument('--dir_out', type=str)
#     parser.add_argument('--sz_space', type=int, nargs='+')
#     parser.add_argument('--plot', action='store_true')
#     parser.add_argument('--benchmark', action='store_true')
#     args = parser.parse_args()

#     # Directory prep
#     dir_res = os.path.join(args.dir_out, 'results')
#     dir_bench = os.path.join(args.dir_out, 'bench')
#     refresh_directory(dir_res)

#     if args.plot:

#         mod_fnames = ['cov-model.pth', 'cov-model-pp.pth']
#         img_prefixes = ['loads', 'loads-pp']

#         for mod_fname, img_prefix in zip(mod_fnames, img_prefixes):
        
#             path_model = os.path.join(dir_out, mod_fname)
#             loads = torch.load(path_model)['loads'].data
#             num_facs = loads.shape[-1]
#             loads = loads.reshape(args.grid_shape + [num_facs])
#             ndim = len(args.grid_shape)
#             dims = list(range(ndim + 1))
#             loads = loads.permute(dims[-1:] + dims[:ndim])
#             for k in range(num_facs):
#                 if ndim == 1:
#                     path = os.path.join(dir_res, f'{img_prefix}_k-{k}.png')
#                     plot_line_for_1d_loads(loads.data, k, path=path)
#                 elif ndim == 2:
#                     path = os.path.join(dir_res, f'{img_prefix}_k-{k}.png')
#                     plot_heatmap_for_2d_loads(loads.data, k, path=path)
#                 elif ndim == 3:
#                     for p in [0.25, 0.5, 0.75]:
#                         z = int(p * args.grid_shape[2])
#                         path = os.path.join(dir_res, f'{img_prefix}_k-{k}_z-{z}.png')
#                         plot_heatmap_for_3d_loads(loads.data, k, z, path=path) 
#                 else:
#                     print(f"`grid_shape` must have length <= 3!")

#     if args.benchmark:

#         path_bench = os.path.join(dir_res, 'bench.csv')
#         rows = []
#         rows += aggregate_benchmarks(dir_bench, 'process_epoch', 'mean')
#         rows += aggregate_benchmarks(dir_bench, 'dataset', 'mean')
#         rows += aggregate_benchmarks(dir_bench, 'other')
#         write_rows_to_csv(path_bench, rows)



