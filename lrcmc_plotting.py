import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.lines import Line2D

from config import load_config
from utils import load_yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--design', type=str)
    parser.add_argument('--n_procs', type=int, nargs='+', help="List of processor counts (e.g., --n_procs 1 4 16)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load design
    path_design = os.path.join(config.group_root, 'designs-lrcmc', f'{args.design}.yml')
    design = load_yaml(path_design)

    # Set up directories
    dir_out = os.path.join(config.group_root, 'out-lrcmc', args.design)
    dir_plots = os.path.join(dir_out, 'plots')
    os.makedirs(dir_plots, exist_ok=True)

    # Plotting setup
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']  # Circle, square, triangle, etc.
    ddp_color = 'tab:blue'
    dsgd_color = 'tab:orange'

    for i, n_proc in enumerate(args.n_procs):
        marker = markers[i % len(markers)]

        # Load CSVs
        path_ddp = os.path.join(dir_out, f'epochs-ddp-{n_proc}.csv')
        path_dsgd = os.path.join(dir_out, f'epochs-dsgd-{n_proc}.csv')
        ddp_df = pd.read_csv(path_ddp)
        dsgd_df = pd.read_csv(path_dsgd)

        # Compute elapsed time
        ddp_df['elapsed_time'] = ddp_df['time'] - ddp_df['time'].iloc[0]
        dsgd_df['elapsed_time'] = dsgd_df['time'] - dsgd_df['time'].iloc[0]

        # Deflate loss by n_procs
        ddp_df['objective'] /= n_proc
        dsgd_df['objective'] /= n_proc

        # Print DSGD epoch time improvement
        ddp_mean = ddp_df['elapsed_time'].mean()
        dsgd_mean = dsgd_df['elapsed_time'].mean()
        print(f"n_proc = {n_proc} | improvement = {dsgd_mean / ddp_mean}")

        # Log10 scale
        ddp_df['log10_objective'] = np.log10(ddp_df['objective'])
        dsgd_df['log10_objective'] = np.log10(dsgd_df['objective'])

        # Plot with marker
        plt.plot(ddp_df['elapsed_time'], ddp_df['log10_objective'],
                 label=f'DDP (n={n_proc})', color=ddp_color, marker=marker, linestyle='-', linewidth=0.5)
        plt.plot(dsgd_df['elapsed_time'], dsgd_df['log10_objective'],
                 label=f'DSGD (n={n_proc})', color=dsgd_color, marker=marker, linestyle='-', linewidth=0.5)

    # Labels and formatting
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Elapsed Time')

    # Build custom legend entries
    method_handles = [
        Line2D([0], [0], color=ddp_color, marker=',', linestyle='-', label='DDP'),
        Line2D([0], [0], color=dsgd_color, marker=',', linestyle='-', label='DSGD')
    ]
    proc_handles = [
        Line2D([0], [0], color='gray', marker=markers[i % len(markers)], linestyle='None', label=f'{n}')
        for i, n in enumerate(args.n_procs)
    ]

    # Add legends to the plot
    first_legend = plt.legend(handles=method_handles, title='Method', loc='upper right')
    plt.gca().add_artist(first_legend)  # Draw first legend manually before second
    plt.legend(handles=proc_handles, title='CPU Count', loc='center right')

    # Custom y-axis labels on original scale
    ax = plt.gca()
    yticks = np.arange(-8, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'$10^{{{int(t)}}}$' for t in yticks])

    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(dir_plots, f'epochs-logloss.png')
    plt.savefig(path)















# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd

# from config import load_config
# from utils import load_yaml


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str)
#     parser.add_argument('--design', type=str)
#     parser.add_argument('--n_procs', type=int, nargs='+')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     # Load design
#     path_design = os.path.join(config.group_root, 'designs-lrcmc', f'{args.design}.yml')
#     design = load_yaml(path_design)

#     # Set up directories
#     dir_out = os.path.join(config.group_root, 'out-lrcmc', args.design)
#     dir_plots = os.path.join(dir_out, 'plots')
#     os.makedirs(dir_plots, exist_ok=True)

#     # Plotting setup
#     plt.figure(figsize=(10, 6))
#     linestyles = ['-', '--', ':', '-.']
#     ddp_color = 'tab:blue'
#     dsgd_color = 'tab:orange'

#     for i, n_proc in enumerate(args.n_procs):

#         # Load CSVs
#         path_ddp = os.path.join(dir_out, f'epochs-ddp-{n_proc}.csv')
#         path_dsgd = os.path.join(dir_out, f'epochs-dsgd-{n_proc}.csv')
#         ddp_df = pd.read_csv(path_ddp)
#         dsgd_df = pd.read_csv(path_dsgd)

#         # Compute elapsed time
#         ddp_df['elapsed_time'] = ddp_df['time'] - ddp_df['time'].iloc[0]
#         dsgd_df['elapsed_time'] = dsgd_df['time'] - dsgd_df['time'].iloc[0]

#         # Deflate loss by n_procs
#         ddp_df['objective'] /= n_proc
#         dsgd_df['objective'] /= n_proc

#         # Log10 scale
#         ddp_df['log10_objective'] = np.log10(ddp_df['objective'])
#         dsgd_df['log10_objective'] = np.log10(dsgd_df['objective'])

#         # Plot
#         linestyle = linestyles[i % len(linestyles)]
#         plt.plot(ddp_df['elapsed_time'], ddp_df['log10_objective'],
#                  label=f'DDP (n={n_proc})', color=ddp_color, linestyle=linestyle)
#         plt.plot(dsgd_df['elapsed_time'], dsgd_df['log10_objective'],
#                  label=f'DSGD (n={n_proc})', color=dsgd_color, linestyle=linestyle)

#     # Labels and formatting
#     plt.xlabel('Elapsed Time (seconds)')
#     plt.ylabel('Training Loss')
#     plt.title('Training Loss vs Elapsed Time')
#     plt.legend()

#     # Custom y-axis labels on original scale
#     ax = plt.gca()
#     yticks = np.arange(-8, 2)
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([f'$10^{{{int(t)}}}$' for t in yticks])

#     plt.grid(True)
#     plt.tight_layout()
#     path = os.path.join(dir_plots, f'epochs-logloss.png')
#     plt.savefig(path)






















# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd

# from config import load_config
# from utils import load_yaml, refresh_directory




# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str)
#     parser.add_argument('--design', type=str)
#     parser.add_argument('--n_procs', type=int)
#     args = parser.parse_args()

#     config = load_config(args.config)

#     # Load design
#     path_design = os.path.join(config.group_root, 'designs-lrcmc', f'{args.design}.yml')
#     design = load_yaml(path_design)

#     # Load CSV files
#     dir_out = os.path.join(config.group_root, 'out-lrcmc', args.design)
#     dir_plots = os.path.join(dir_out, 'plots')
#     path_ddp = os.path.join(dir_out, f'epochs-ddp-{args.n_procs}.csv')
#     path_dsgd = os.path.join(dir_out, f'epochs-dsgd-{args.n_procs}.csv')
#     ddp_df = pd.read_csv(path_ddp)
#     dsgd_df = pd.read_csv(path_dsgd)
#     os.makedirs(dir_plots, exist_ok=True)

#     # Compute elapsed time
#     ddp_start_time = ddp_df['time'].iloc[0]
#     dsgd_start_time = dsgd_df['time'].iloc[0]
#     ddp_df['elapsed_time'] = ddp_df['time'] - ddp_start_time
#     dsgd_df['elapsed_time'] = dsgd_df['time'] - dsgd_start_time

#     # Deflate loss by n_procs
#     ddp_df['objective'] /= args.n_procs
#     dsgd_df['objective'] /= args.n_procs

#     # Compute log-objective
#     ddp_df['log10_objective'] = np.log10(ddp_df['objective'])
#     dsgd_df['log10_objective'] = np.log10(dsgd_df['objective'])

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(ddp_df['elapsed_time'], ddp_df['log10_objective'], label='DDP', marker='o')
#     plt.plot(dsgd_df['elapsed_time'], dsgd_df['log10_objective'], label='DSGD', marker='s')

#     # Labels and legend
#     plt.xlabel('Elapsed Time (seconds)')
#     plt.ylabel('Training Loss')
#     plt.title('Training Loss vs Elapsed Time')
#     plt.legend()

#     # Custom y-axis: log10 scale with original scale labels
#     ax = plt.gca()
#     yticks = np.arange(-8, 2)  # Adjust range as needed
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([f'$10^{{{int(tick)}}}$' for tick in yticks])

#     plt.grid(True)
#     plt.tight_layout()
#     path = os.path.join(dir_plots, f'epochs-{args.n_procs}-logloss.png')
#     plt.savefig(path)




