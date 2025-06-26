import argparse
import numpy as np
import os
import pandas as pd
import torch

from utils.benchmarking import aggregate_benchmarks
from config import load_config
from utils import (
    l2_norm,
    load_yaml,
    multiply_list,
    procrustes_rotation,
    refresh_directory, 
    write_rows_to_csv,
)
from utils.plotting import (
    plot_line_for_1d_loads,
    plot_heatmap_for_2d_loads,
    plot_heatmap_for_3d_loads,
    plot_n_arrays,
    plot_side_by_side_heatmaps,
)

# NOTE: (Script Outline)
#   - Required metrics (computed in this script)
#       * LE1: ||G - G_hat|| / ||G| ; from L and L_hat
#       * LE2: train_err and time per epoch; from /bench 
#       * FE1: ||F - F_hat|| / F; from F and F_hat
#       * FE2: Same
#       * FE3: Same


def assign_load_config(row):
    if row['load_scheme'].startswith('Bump'):
        prefix = 'BUMP'
    elif row['load_scheme'].startswith('Net'):
        prefix = 'NET'
    else: 
        raise Exception("Invalid loading scheme!")
    return f"{prefix}_{row['n_facs']}"



# def assign_err_config(row):

#     if row['err_scheme'].startswith('GaussProc_Bump'):
#         prefix = 'BE'
#     elif row['load_scheme'].startswith('GaussProc_BSpline'):
#         prefix = 'BS'
#     else: 
#         raise Exception("Invalid loading scheme!")
    
#     if row['delta'] == 0.05:
#         suffix = '05'
#     elif row['delta'] == 0.1:
#         suffix = '10'
#     else: 
#         raise Exception("Invalid delta!")

#     return prefix + suffix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--dir_designs', type=str)
    parser.add_argument('--designs', type=str, nargs='+')
    parser.add_argument(
        '--sim_type', type=str, required=True,
        choices=['load', 'fac']
    )
    parser.add_argument(
        '--est_methods_load', type=str, nargs='+',
        choices=['ffa', 'mc', 'mcs', 'ica', 'icas'],
        default=['ffa', 'mc', 'mcs', 'ica', 'icas']
    )
    parser.add_argument(
        '--est_methods_fac', type=str, nargs='+',
        choices=['pls', 'rbels'],
        default=['pls', 'rbels'],
    )
    args = parser.parse_args()
    config = load_config(args.config)

    # Gather relevant design information
    designs = {}
    for des_id in args.designs: 
        designs[des_id] = load_yaml(
            os.path.join(args.dir_designs, f'{des_id}.yml')
        )
        designs[des_id]['sim_ids'] = os.listdir(
            os.path.join(args.dir_designs, des_id)
        )


    if args.sim_type == 'load':

        # TODO: Handle NA entries

        # Estimation method to model name map
        method_to_model = {
            'ffa': 'model-dssgd-full-smooth-varimax-shrink.pth',
            'mc': 'model-dssgd-full.pth',
            'mcs': 'model-dssgd-full-smooth.pth',
            'ica': 'ica-space_split-full.npy', # TODO: (1) update to non-smoothed version, (2) save as LowRankCovariance Model, (3) verify scale
            'icas': 'icas-space_split-full.npy',
        }

        # Build dataframe of simulation results from all designs
        columns = [
            'des_id', 'sim_id', 'rep_id',
            'est_method',
            'load_scheme', 'n_facs',
            'delta', 'regime', 'n_sub'
        ]
        df = pd.DataFrame(columns=columns)
        for des_id, design in designs.items(): 
            for sim_id in design['sim_ids']:
                print(f"{des_id} | {sim_id}")

                for r in range(design['n_reps']):
                    
                    # Load repetition
                    rep = load_yaml(os.path.join(
                        args.dir_designs,
                        des_id, 
                        sim_id, 
                        f'rep-{r}.yml'
                    ))

                    # Read true loadings and compute true global covariance
                    loads = torch.load(os.path.join(rep['dir_out'], 'loads.pt'))
                    n_space = multiply_list(loads.shape[1:])
                    loads = loads.reshape(loads.size(0), n_space)
                    g = loads.t() @ loads
                    g /= torch.norm(g)

                    # Compute estimation error for each method then add row to dataframe
                    add_rows = True
                    row_list = []
                    for method in args.est_methods_load:

                        try: 
                            path = os.path.join(rep['dir_out'], method_to_model[method])
                            if method in ['ica', 'icas']: 
                                loads = torch.tensor(np.load(path).reshape(n_space, rep['n_facs'][0]))                     
                            else: 
                                loads = torch.load(path)['loads'].data
                            
                        except FileNotFoundError:
                            print(f"File does not exist: {path}")
                            add_rows = False
                            break

                        g_hat = loads @ loads.t()
                        g_hat /= torch.norm(g_hat)
                        err = torch.norm(g_hat - g) / torch.norm(g)

                        row_list.append({
                            'des_id': des_id, 
                            'sim_id': sim_id, 
                            'rep_id': f'rep-{r}',
                            'est_method': method,
                            'load_scheme': rep['load_scheme'],
                            'n_facs': rep['n_facs'][0],
                            'delta': rep['delta'],
                            'regime': rep['regime'],
                            'n_sub': rep['n_sub'],
                            'err': err.item()
                        })


                    if add_rows:
                        df = pd.concat([df, pd.DataFrame(row_list)])

        # Assign load config (Ex: BUMP_2)
        # df['load_config'] = df.apply(assign_load_config, axis=1)

        # Assign relative errors
        df['err_ffa'] = pd.merge(
            df, df[df.est_method == 'ffa'], 
            how='left',
            left_on=['des_id', 'sim_id', 'rep_id'],
            right_on=['des_id', 'sim_id', 'rep_id']
        )['err_y'].to_list()
        df['rel_err'] = df['err'] / df['err_ffa']

        # Aggregate errors and relative errors
        df_agg = df.groupby([
            'load_scheme', 'n_facs', 'delta', 'regime', 'n_sub', 'est_method'
        ]).agg(
            n_reps=('err', 'count'), 
            mean_err=('err', 'mean'),
            std_err=('err', 'std'),
            q10_err=('err', lambda x: x.quantile(0.1)),
            q50_err=('err', lambda x: x.quantile(0.5)),
            q90_err=('err', lambda x: x.quantile(0.9)),
            mean_rel_err=('rel_err', 'mean'),
            std_rel_err=('rel_err', 'std'),
            q10_rel_err=('rel_err', lambda x: x.quantile(0.1)),
            q50_rel_err=('rel_err', lambda x: x.quantile(0.5)),
            q90_rel_err=('rel_err', lambda x: x.quantile(0.9))
        ).reset_index()
        print(df_agg)

        # Save aggregated dataframe
        path = os.path.join(config.root, 'results', 'sim-load.csv')
        df_agg.to_csv(path, index=False)

    # NOTE: 
    #   - ICA and ICAS have lower-variance errors
    #   - ICA and ICAS have higher-variance relative errors


    if args.sim_type == 'fac':

        path_mask_to_n_space = {
            '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_10-10.pt': 10,
            '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_20-20.pt': 20,
            '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_30-30.pt': 30,
            '/storage/home/kms8227/work/ffa-p2-priv/masks/mask_40-40_40-40.pt': 40, 
        }

        columns = [
            'des_id', 'sim_id', 'rep_id', 
            'n', 'k',
            'est_method',
            'fac_cov',
            'n_space', 'delta'
        ]
        df = pd.DataFrame(columns=columns)
        for des_id, design in designs.items():
            for sim_id in design['sim_ids']:
                print(f"{des_id} | {sim_id}")

                for r in range(design['n_reps']):

                    # Load repetition
                    rep = load_yaml(os.path.join(
                        args.dir_designs,
                        des_id, 
                        sim_id, 
                        f'rep-{r}.yml'
                    ))

                    # Load factors
                    facs = torch.load(os.path.join(rep['dir_out'], 'facs.pt'))

                    # Compute estimation error each method
                    for method in args.est_methods_fac:
                        
                        path = os.path.join(
                            rep['dir_out'], 
                            f'facs_split-full_m-{method}_rot-varimax_reg-1.pt'
                        )
                        est_facs = torch.load(path)
                        for n in range(rep['n_sub']):
                            
                            # Normalize subject's factors
                            facs_n = facs[n] / torch.norm(facs[n])
                            est_facs_n = est_facs[n] / torch.norm(est_facs[n])

                            # Apply Procrustes rotation to ensure alignment
                            rot_mat = procrustes_rotation(est_facs_n, facs_n)
                            est_facs_n = rot_mat @ est_facs_n
                                
                            # Compute error
                            for k in range(rep['n_facs'][0]):
                                err = torch.norm(est_facs_n[k] - facs_n[k]) / torch.norm(facs_n[k])
                                row = {
                                    'des_id': des_id,
                                    'sim_id': sim_id,
                                    'rep_id': r,
                                    'n': n,
                                    'k': k,
                                    'est_method': method,
                                    'fac_cov': 'oblique' if des_id.endswith('-obl') else 'orthogonal',
                                    'n_space': path_mask_to_n_space[rep['path_mask']],
                                    'delta': rep['delta'],
                                    'err': err.item()
                                }
                                df = pd.concat([df, pd.DataFrame([row])])

        # Aggregate
        df_agg = df.groupby([
            'n_space', 'delta', 'est_method', 'fac_cov'
        ]).agg(
            mean_err=('err', 'mean'),
            std_err=('err', 'std'),
            q10_err=('err', lambda x: x.quantile(0.1)),
            q50_err=('err', lambda x: x.quantile(0.5)),
            q90_err=('err', lambda x: x.quantile(0.9))
        ).reset_index()
        print(df_agg)

        # Save aggregated dataframe
        path = os.path.join(config.root, 'results', 'sim-fac.csv')
        df_agg.to_csv(path, index=False)

            
