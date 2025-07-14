import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import torch

from utils import load_yaml


# ---------- Subspace Estimation ---------- #

# path = os.path.join('results', 'sim-load.csv')
# df = pd.read_csv(path)

# # Define "labels"
# # df['label'] = df.apply(lambda row: f"({row['delta']},{row['regime']},{row['n_sub']})", axis=1)
# df['label'] = df.apply(lambda row: f"({row['regime']},{row['delta']},{row['n_sub']})", axis=1)

# # Set dictionaries
# shapes = {
#     'ffa': 's', 
#     'icas': 'x', 
#     'ica': 'X', 
#     'mc': '^', 
#     'mcs': 'o'
# }
# colors = {
#     'ffa': 'tab:green', 
#     'icas': 'tab:blue', 
#     'ica': 'tab:orange', 
#     'mc': 'tab:red', 
#     'mcs': 'tab:purple'
# }
# scheme_map = {
#     'BumpScheme2D': 'BI',
#     'NetScheme2D': 'NET'
# }

# # Set up the FacetGrid
# g = sns.FacetGrid(df, row='load_scheme', col='n_facs', height=5, sharex=True, sharey=True)


# def plot_method(data, metric, **kwargs):

#     metric_map = {

#         'err': {
#             'methods_order': ['tffa', 'mcs', 'mc', 'icas', 'ica'],
#             'vline': 0,
#             'xlabel': 'Global Covariance Reconstruction Error',
#         },

#         'rel_err': {
#             'methods_order': ['mcs', 'mc', 'icas', 'ica'],
#             'vline': 1,
#             'xlabel': 'Error Relative to TFFA',
#         }

#     }

#     def parse_label(label):
#         nums = re.findall(r"[-+]?\d*\.\d+|\d+", label)
#         return tuple(float(x) for x in nums)

#     # Sort label
#     labels = data['label'].unique()
#     labels_sorted = sorted(labels, key=parse_label, reverse=True)
#     label_to_y = {label: i for i, label in enumerate(labels_sorted)}
#     data['y'] = data['label'].map(label_to_y)

#     # Get ordered methods
#     methods_order = metric_map[metric]['methods_order']
#     for method in methods_order:
#         if method not in data['est_method'].unique():
#             methods_order.remove(method)

#     for method in methods_order:
#         method_data = data[data['est_method'] == method]
#         # y = method_data['y'] + 0.075 * (list(shapes.keys()).index(method) - 2)  # dodge
#         y = method_data['y'] - 0.075 * (methods_order.index(method) - 2)  # dodge
#         # q10 = method_data[f'q10_{metric}'].values
#         # q50 = method_data[f'q50_{metric}'].values
#         # q90 = method_data[f'q90_{metric}'].values
#         # lower = q50 - q10
#         # upper = q90 - q50
#         mean = method_data[f'mean_{metric}'].values
#         std = method_data[f'std_{metric}'].values
#         lower = 2 * std
#         upper = 2 * std
#         plt.errorbar(
#             x=mean,#q50,
#             y=y,
#             xerr=np.row_stack([lower, upper]),
#             fmt=shapes[method],
#             color=colors[method],
#             label=method,
#             capsize=3,
#             markersize=4,
#             linestyle='None'
#         )

#     # Add vertical line
#     # plt.axvline(x=0, color='black', linestyle='dotted')
#     plt.axvline(x=metric_map[metric]['vline'], color='black', linestyle='dotted')
#     plt.yticks(list(label_to_y.values()), list(label_to_y.keys()))
#     plt.xlabel(metric_map[metric]['xlabel'])
#     plt.ylabel("(regime,\u03B4,n)")



# # plot_method(df[(df['load_scheme'] == 'BUMP') & (df['n_facs'] == 2)], os.path.join('results', 'test.png'))
# g.map_dataframe(plot_method, metric='rel_err')

# # Update facet titles
# for ax in g.axes.flat:
#     title = ax.get_title()
#     # Example title: "load_scheme = BumpScheme2D | n_facs = 4"
#     match = re.search(r"load_scheme\s*=\s*(\w+)\s*\|\s*n_facs\s*=\s*(\d+)", title)
#     if match:
#         scheme, k = match.groups()
#         scheme_code = scheme_map.get(scheme, scheme)
#         ax.set_title(f"{scheme_code} | K = {k}")

# # Legend (only once)
# handles, labels = plt.gca().get_legend_handles_labels()
# methods_order = ['ffa', 'mcs', 'mc', 'icas', 'ica']
# label_to_handle = dict(zip(labels, handles))
# ordered_handles = [label_to_handle[label] for label in methods_order if label in label_to_handle]
# ordered_labels = [label.upper() for label in methods_order if label in label_to_handle]
# plt.legend(ordered_handles, ordered_labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
# # plt.legend(handles, labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Save
# path = os.path.join('results', 'load-grid.png')
# plt.tight_layout()
# plt.savefig(path)


# ---------- Subspace Expression ---------- #

def plot_2d_load(load, max_mag=None):
    """Return a plot object (figure and axis) for a 2D loading."""
    fig, ax = plt.subplots()

    # Determine range if not provided
    max_mag = max_mag if max_mag else np.max(np.abs(load))

    # Define colormap: 'bwr' for brain, 'lightgray' for non-brain
    cmap = plt.cm.bwr
    cmap.set_bad(color='lightgray')

    # Plot heatmap
    im = ax.imshow(load, cmap=cmap, origin='lower', vmin=-max_mag, vmax=max_mag)

    # Keep ticks but remove tick labels
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig, ax, im, cmap


# Globals
design_ids = [

    # 'sim-pilot-9_ktrue-4_kest-12_faccov-orth',
    # 'sim-pilot-9_ktrue-4_kest-16_faccov-orth',
    # 'sim-pilot-9_ktrue-4_kest-20_faccov-orth',

    # 'sim-pilot-9_ktrue-4_kest-12_faccov-obl2',
    # 'sim-pilot-9_ktrue-4_kest-16_faccov-obl2',
    # 'sim-pilot-9_ktrue-4_kest-20_faccov-obl2',

    # 'sim-pilot-11_pg-20_delta-5',
    # 'sim-pilot-11_pg-15_delta-5',
    # 'sim-pilot-11_pg-10_delta-5',
    # 'sim-pilot-11_pg-5_delta-5',

    # 'sim-pilot-14_ktrue-8_kest-8_faccov-orth',
    # 'sim-pilot-14_ktrue-8_kest-8_faccov-obl',
    # 'sim-pilot-14_ktrue-8_kest-25_faccov-orth',
    # 'sim-pilot-14_ktrue-8_kest-25_faccov-obl',

    # 'sim-pilot-15_ktrue-8_kest-25_faccov-obl',

    # 'sim-test-0_ktrue-8_kest-8_faccov-orth'

    # 'sim-exp-2_ktrue-8_kest-25_faccov-orth',
    # 'sim-exp-2_ktrue-8_kest-25_faccov-obl',

    # 'sim-defense_ktrue-2_kest-2_faccov-orth'

    'sim-exp-3_1',  # K = 8
    'sim-exp-3_2',

    # 'sim-exp-3_3',  # K = 25
    # 'sim-exp-3_4',


]

n_sims = 1  # sim = 0
n_reps = 10
n_rows = 2
n_cols = 4

method_to_file = {

    # 'init': 'init-loads-full.pt',

    # 'mc': 'model-dssgd-full.pth',
    # 'mcs': 'model-dssgd-full-smooth.pth',

    # 'mc-or': 'model-dssgd-full-varimax.pth',
    # 'mc-ob': 'model-dssgd-full-quartimin.pth',

    # 'mcs-or': 'model-dssgd-full-smooth-varimax.pth',
    # 'mcs-ob': 'model-dssgd-full-smooth-quartimin.pth',

    'ffa-or': 'model-dssgd-full-varimax-smooth-shrink.pth',
    'ffa-ob': 'model-dssgd-full-quartimin-smooth-shrink.pth',

    # 'ica': 'ica-space_split-full.npy',
    'icas': 'icas-space_split-full.npy',

}

for design_id in design_ids: 

    for sim in range(n_sims):

        for rep in range(n_reps):

            path_design = os.path.join(
                '/storage/group/kms8227/default/ffa-p2-priv/designs',
                design_id, f'sim-{sim}', f'rep-{rep}.yml', 
            )
            design = load_yaml(path_design)
            n_facs_est = design['n_facs'][1]
            sz_space = design['sz_space']
            dir_out = os.path.join(
                '/storage/group/kms8227/default/ffa-p2-priv/out',
                design_id, f'sim-{sim}', f'rep-{rep}'
            )

            for method, file in method_to_file.items():

                # Read loads
                try: 
                    path = os.path.join(dir_out, file)
                    if path.endswith('.npy'):
                        loads = np.transpose(np.load(path).reshape(*sz_space, n_facs_est), (2, 0, 1))
                        loads = torch.tensor(loads)
                    elif path.endswith('.pt'):
                        loads = torch.load(path).t().reshape(n_facs_est, *sz_space)
                    else: 
                        loads = torch.load(path)['loads'].data.t().reshape(n_facs_est, *sz_space)
                except FileNotFoundError: 
                    print(f"Path does not exist: {path}")
                    continue


                max_mag = torch.max(torch.abs(loads)).item()
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
                axes = axes.flatten()

                for k in range(n_facs_est):

                    # Positive loadings
                    loads[k] *= torch.sign(torch.sum(loads[k]))

                    _, _, im, cmap = plot_2d_load(loads[k], max_mag)

                    # Copy the plot to the subplot
                    axes[k].imshow(im.get_array(), cmap=cmap, origin='lower', vmin=im.get_clim()[0], vmax=im.get_clim()[1])
                    axes[k].set_title(f'k = {k+1}', fontsize=24)

                # Add a common colorbar
                # cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Position colorbar
                # fig.colorbar(im, cax=cbar_ax, label=None)

                # Adjust layout and save
                plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
                path = os.path.join(dir_out, 'results', f'loads_m-{method}.png')
                fig.savefig(path, dpi=300)  # Save to PNG
                plt.close()








# # ---------- Factor Score Estimation ---------- #

# path = os.path.join('results', 'sim-fac.csv')
# df = pd.read_csv(path)

# # Define "labels"
# df['label'] = df.apply(lambda row: f"({row['n_space']},{row['delta']})", axis=1)

# # Set shape and color dictionaries
# shapes = {
#     'pls': 's', 
#     'rbels': 'x', 
# }
# colors = {
#     'pls': 'tab:blue', 
#     'rbels': 'tab:orange', 
# }

# # Set up the FacetGrid
# g = sns.FacetGrid(df, col='fac_cov', height=5, sharex=True, sharey=True)

# def plot_method(data, metric, **kwargs):
#     labels = data['label'].unique()
#     label_to_y = {label: i for i, label in enumerate(reversed(sorted(labels)))}
#     data['y'] = data['label'].map(label_to_y)

#     for method in data['est_method'].unique():
#         method_data = data[data['est_method'] == method]
#         y = method_data['y'] + 0.15 * (list(shapes.keys()).index(method) - 2)  # dodge
#         q10 = method_data[f'q10_{metric}'].values
#         q50 = method_data[f'q50_{metric}'].values
#         q90 = method_data[f'q90_{metric}'].values
#         lower = q50 - q10
#         upper = q90 - q50
#         plt.errorbar(
#             x=q50,
#             y=y,
#             xerr=np.row_stack([lower, upper]),
#             fmt=shapes[method],
#             color=colors[method],
#             label=method,
#             capsize=3,
#             markersize=6,
#             linestyle='None'
#         )

#     # Add vertical line
#     plt.axvline(x=0, color='black', linestyle='dotted')
#     plt.yticks(list(label_to_y.values()), list(label_to_y.keys()))
#     plt.xlabel("Factor Reconstruction Error")
#     plt.ylabel("(h,\u03B4)")

# # plot_method(df[(df['load_scheme'] == 'BUMP') & (df['n_facs'] == 2)], os.path.join('results', 'test.png'))
# g.map_dataframe(plot_method, metric='err')
# for ax in g.axes.flat:
#     ax.set_title("")

# # Create legend
# label_map = {'pls': 'PWLS', 'rbels': 'FOSR'}
# desired_order = ['FOSR', 'PWLS']
# handles, labels = plt.gca().get_legend_handles_labels()
# mapped_labels = [label_map.get(label, label) for label in labels]
# label_to_handle = dict(zip(mapped_labels, handles))
# ordered_labels = [label for label in desired_order if label in label_to_handle]
# ordered_handles = [label_to_handle[label] for label in ordered_labels]
# plt.legend(ordered_handles, ordered_labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Save
# path = os.path.join('results', 'fac-grid.png')
# plt.tight_layout()
# plt.savefig(path)





# ---------- Loading Schemes --------- #


# def plot_2d_load(load, max_mag=None):
#     """Return a plot object (figure and axis) for a 2D loading."""
#     fig, ax = plt.subplots()

#     # Determine range if not provided
#     max_mag = max_mag if max_mag else np.max(np.abs(load))

#     # Define colormap: 'bwr' for brain, 'lightgray' for non-brain
#     cmap = plt.cm.bwr
#     cmap.set_bad(color='lightgray')

#     # Plot heatmap
#     im = ax.imshow(load, cmap=cmap, origin='lower', vmin=-max_mag, vmax=max_mag)

#     # Keep ticks but remove tick labels
#     ax.set_xticks(ax.get_xticks())
#     ax.set_yticks(ax.get_yticks())
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])

#     return fig, ax, im, cmap


# # Globals
# sz_space = [30, 30]
# analysis = 'sim-est-2-1_regime-1_loads-BL_K-2' # sim-est-2-1_regime-1_loads-NL_K-4, sim-est-2-1_regime-1_loads-BL_K-4, sim-exp-0_ktrue-8_kest-8_faccov-orth
# n_rows = 1
# n_cols = 2
# dir_out = f'/storage/group/kms8227/default/ffa-p2-priv/out/{analysis}/sim-0/rep-0'

# # Read in loadings
# path = os.path.join(dir_out, 'loads.pt')
# loads = torch.load(path)
# n_facs = loads.shape[0]
# loads = loads.reshape(n_facs, *sz_space)
# # loads = loads.permute(0, 2, 1)
# for k in range(n_facs):
#     loads[k] /= torch.norm(loads[k])

# # Set up axes
# max_mag = torch.max(torch.abs(loads)).item()
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
# axes = axes.flatten()

# loads[1] *= 0.6

# for k in range(n_facs):

#     _, _, im, cmap = plot_2d_load(loads[k].T, max_mag)

#     # Copy the plot to the subplot
#     axes[k].imshow(im.get_array(), cmap=cmap, origin='lower', vmin=im.get_clim()[0], vmax=im.get_clim()[1])
#     axes[k].set_title(f'k = {k+1}', fontsize=24)

# # Add a common colorbar
# # cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Position colorbar
# # fig.colorbar(im, cax=cbar_ax, label=None)

# # Adjust layout and save
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
# # path = os.path.join('results', f'loads-scheme_{analysis}.png')
# path = os.path.join('defense-plots', f'demo-loads-true.png')
# fig.savefig(path, dpi=300)  # Save to PNG
# plt.close()


# ---------- Factor Score Estimates ---------- #


# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# from matplotlib import colors as mcolors

# def plot_fse(tensor_list, title=None, path_out=None):
#     """
#     Plots a list of PyTorch tensors as individual lines on a single plot.

#     Parameters:
#     - tensor_list (list of torch.Tensor): List of length-n, where each element is a length-J 1D tensor.
#     - title (str): Title of the plot.
#     - path_out (str or None): If provided, saves plot to this path. Otherwise, displays it.
#     """
#     n = len(tensor_list)
#     if n == 0:
#         raise ValueError("tensor_list must contain at least one tensor.")
    
#     J = tensor_list[0].numel()
#     for i, tensor in enumerate(tensor_list):
#         if tensor.numel() != J:
#             raise ValueError(f"Tensor at index {i} does not have length {J}.")

#     x = np.arange(1, J + 1)

#     # Use a list of distinct Tableau colors
#     tableau_colors = list(mcolors.TABLEAU_COLORS.values())
#     if n > len(tableau_colors):
#         raise ValueError(f"Only {len(tableau_colors)} distinct colors available, but got {n} tensors.")

#     plt.figure(figsize=(10, 6))
#     for i, tensor in enumerate(tensor_list):
#         y = tensor.detach().cpu().numpy()
#         plt.plot(x, y, label=f"Subject {i + 1}", color=tableau_colors[i])
    
#     plt.xlabel("Time", fontsize=18)
#     if title:
#         plt.title(title, fontsize=24)
#     plt.legend(loc="upper right", fontsize=16)
#     plt.tight_layout()

#     if path_out: 
#         plt.savefig(path_out)
#     else: 
#         plt.show()

    


# # Globals
# design = 'sim-defense_ktrue-2_kest-2_faccov-orth'
# sim = 0
# rep = 0
# n_facs = 2
# n_subs = 2
# est_method = 'rbels'
# rot_method = 'varimax'
# dir_out = os.path.join(
#     '/storage/group/kms8227/default/ffa-p2-priv/out',
#     design, f'sim-{sim}', f'rep-{rep}'
# )

# # Read in factors
# path = os.path.join(dir_out, f'facs_split-full_m-{est_method}_rot-{rot_method}_reg-3.pt')
# facs = torch.load(path)
# print(facs.shape)

# # Plot
# for k in range(n_facs):
#     fac_list = []
#     for i in range(n_subs):
#         fac_list.append(facs[i,k,:])
#     path = os.path.join(dir_out, 'results', f'demo-fse-{k}.png')
#     plot_fse(fac_list, title=f"k = {k+1}", path_out=path)



