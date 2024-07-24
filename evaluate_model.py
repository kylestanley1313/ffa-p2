import argparse
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
        prefix = 'BL'
    elif row['load_scheme'].startswith('Net'):
        prefix = 'NET'
    else: 
        raise Exception("Invalid loading scheme!")
    return prefix + str(row['n_facs'])


def assign_err_config(row):

    if row['err_scheme'].startswith('GaussProc_Bump'):
        prefix = 'BE'
    elif row['load_scheme'].startswith('GaussProc_BSpline'):
        prefix = 'BS'
    else: 
        raise Exception("Invalid loading scheme!")
    
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
    parser.add_argument('--designs', type=str, nargs='+')
    parser.add_argument(
        '--sim_type', type=str, required=True,
        choices=['load_err', 'load_bench', 'fac']
    )
    parser.add_argument(
        '--le_methods', type=str, nargs='+',
        choices=['lbfgs', 'dsgd', 'dssgd'],
        default=['lbfgs', 'dsgd', 'dssgd']
    )
    parser.add_argument(
        '--fse_methods', type=str, nargs='+', 
        choices=['pls', 'pgls', 'rbels', 'rbegls'],
        default=['pls', 'pgls', 'rbels', 'rbegls']
    )
    parser.add_argument(
        '--fse_regimes', type=int, nargs='+',
        choices=[1, 2, 3],
        default=[1, 2, 3]
    )
    args = parser.parse_args()
    config = load_config(args.config)

    # Gather relevant design information
    designs = {}
    for des_id in args.designs: 
        designs[des_id] = load_yaml(
            os.path.join(config.root, 'designs', f'{des_id}.yml')
        )
        designs[des_id]['dir_design'] = os.path.join(config.root, 'designs', des_id)
        designs[des_id]['dir_out'] = os.path.join(config.root, 'out', des_id)
        designs[des_id]['sim_ids'] = os.listdir(designs[des_id]['dir_out'])


    if args.sim_type == 'load_err':

        # Build dataframe of simulation results from all designs
        columns = [
            'des_id', 'sim_id', 'rep_id', 'est_method',
            'load_scheme', 'n_facs', 'err_scheme', 'delta',  # figure columns
            'prop_glob', 'fac_kern_length', 'n_time',  # figure rows
            'err'
        ]
        df = pd.DataFrame(columns=columns)
        for des_id, design in designs.items(): 
            for sim_id in design['sim_ids']:

                # Read true loadings
                dir_sim = os.path.join(config.root, 'designs', des_id, sim_id)
                dir_out_sim = os.path.join(config.root, 'out', des_id, sim_id)
                loads = torch.load(os.path.join(dir_out_sim, 'loads.pt'))
                
                # Compute true global covariance
                n_space = multiply_list(loads.shape[1:])
                loads = loads.reshape(loads.size(0), n_space)
                g = loads.t() @ loads

                for r in range(design['n_reps']):
                    
                    # Load repetition
                    rep = load_yaml(os.path.join(dir_sim, f'rep-{r}.yml'))

                    # Compute estimation error for each method then add row to dataframe
                    for method in args.le_methods:

                        path = os.path.join(rep['dir_out'], f'model-{method}-full.pth')
                        est_loads = torch.load(path)['loads'].data
                        g_hat = est_loads @ est_loads.t()
                        err = torch.norm(g_hat - g) / torch.norm(g)

                        row = {
                            'des_id': des_id,
                            'sim_id': sim_id,
                            'rep_id': f'rep-{r}',
                            'est_method': method,
                            'load_scheme': rep['load_scheme'],
                            'n_facs': rep['n_facs'],
                            'err_scheme': rep['err_scheme'],
                            'delta': rep['delta'],
                            'prop_glob': rep['prop_global'],
                            'fac_kern_length': rep['factor_kernel_length'],
                            'n_time': rep['n_time'],
                            'err': err.item()
                        }
                        df = pd.concat([df, pd.DataFrame([row])])

        # Assign configs
        df['load_config'] = df.apply(assign_load_config, axis=1)
        df['err_config'] = df.apply(assign_err_config, axis=1)

        # Assign errors relative to lbfgs
        df['err_lbfgs'] = pd.merge(
            df, df[df.est_method == 'lbfgs'], 
            how='left',
            left_on=['des_id', 'sim_id', 'rep_id'],
            right_on=['des_id', 'sim_id', 'rep_id']
        )['err_y'].to_list()
        df['rel_err'] = df['err'] / df['err_lbfgs']

        df_agg = df.groupby([
            'est_method', 
            'load_config', 'err_config', 
            'prop_glob', 'fac_kern_length', 'n_time']).agg(
                mean_err=('rel_err', 'mean'),
                std_err=('rel_err', 'std')
            ).reset_index()
        print(df_agg)


    if args.sim_type == 'load_bench':

        df_list = []
        for des_id, design in designs.items():
            for sim_id in design['sim_ids']:
                dir_sim = os.path.join(config.root, 'designs', des_id, sim_id)

                for r in range(design['n_reps']):
                    rep = load_yaml(os.path.join(dir_sim, f'rep-{r}.yml'))

                    for method in args.le_methods: 
                        path = os.path.join(rep['dir_out'], 'bench', f'epochs-{method}.csv')
                        df = pd.read_csv(path)
                        df['des_id'] = des_id
                        df['sim_id'] = sim_id
                        df['rep_id'] = r
                        df['est_method'] = method
                        df['world_size'] = rep['world_size_est']
                        df['n_vars'] = multiply_list(rep['sz_space'])
                        df_list.append(df)
        df = pd.concat(df_list)
        print(df.shape)


    if args.sim_type == 'fac':

        # Build dataframe of simulation results from all designs
        columns = [
            'des_id', 'sim_id', 'rep_id', 'est_method', 'regime',
            'load_scheme', 'n_facs', 'err_scheme', 'delta',  # figure columns
            'prop_glob', 'fac_kern_length', 'n_time',  # figure rows
            'err'
        ]
        df = pd.DataFrame(columns=columns)
        for des_id, design in designs.items():
            for sim_id in design['sim_ids']:

                # Read true factors
                dir_sim = os.path.join(config.root, 'designs', des_id, sim_id)

                for r in range(design['n_reps']):

                    # Load repetition
                    rep = load_yaml(os.path.join(dir_sim, f'rep-{r}.yml'))

                    # Load factors
                    facs = torch.load(rep['dir_out_rep'], 'facs.pt')

                    # Compute estimation error each method
                    for method in args.fse_methods:
                        for regime in args.fse_regimes: 

                            # Read estimated factors
                            path = os.path.join(
                                rep['dir_out'], 
                                f'facs_full_r{regime}_{method}.pt'
                            )
                            est_facs = torch.load(path)
                            
                            if regime == 3: 
                                # Rotate estimated factors towards true factors.
                                # This is required since regime 3 uses 
                                # estimated, not true, loadings. 
                                rot_mat = procrustes_rotation(est_facs, facs)
                                est_facs = rot_mat @ est_facs

                            err = torch.sum((est_facs - facs) ** 2)
                            err = err / torch.norm(facs)

                            row = {
                                'des_id': des_id,
                                'sim_id': sim_id,
                                'rep_id': f'rep-{r}',
                                'est_method': method,
                                'regime': regime,
                                'load_scheme': rep['load_scheme'],
                                'n_facs': rep['n_facs'],
                                'err_scheme': rep['err_scheme'],
                                'delta': rep['delta'],
                                'prop_glob': rep['prop_global'],
                                'fac_kern_length': rep['factor_kernel_length'],
                                'n_time': rep['n_time'],
                                'err': err.item()
                            }
                            df = pd.concat([df, pd.DataFrame([row])])

        df_agg = df.groupby([
            'est_method', 'regime',
            'load_scheme', 'n_facs', 'err_scheme', 'delta',
            'prop_glob', 'fac_kern_length', 'n_time']).agg(
                mean_err=('err', 'mean'),
                std_err=('err', 'std')
            ).reset_index()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_agg)


    



