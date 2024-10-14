import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ---------- PLOT LOADINGS ---------- #

def plot_line_for_1d_loads(tensor, k, path=None):
    vals = tensor[k].numpy()
    sns.lineplot(x=range(len(vals)), y=vals)
    plt.title(f"Line Plot of tensor[{k}]")
    plt.xlabel("x")
    plt.ylabel("vals")
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_heatmap_for_2d_loads(tensor, k, path=None):
    slice_tensor = tensor[k, :, :]
    sns.heatmap(slice_tensor.numpy(), cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Heatmap of tensor[{k}, :, :]")
    plt.xlabel("x")
    plt.ylabel("y")
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_heatmap_for_3d_loads(tensor, k, z, path=None):
    slice_tensor = tensor[k, :, :, z]
    sns.heatmap(slice_tensor.numpy(), cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Heatmap of tensor[{k}, :, :, {z}]")
    plt.xlabel("x")
    plt.ylabel("y")
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()



# ---------- PLOT NUMPY ARRAYS ---------- #

def plot_array(arr):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(arr)), arr)
    plt.title('Array')
    plt.xlabel('idx')
    plt.ylabel('val')
    plt.grid(True)
    plt.show()


def rgb_to_hex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)


def get_random_color(seed): 
    np.random.seed(seed)
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return rgb_to_hex(r, g, b)


def plot_n_arrays(arrays, seed=1234, path=None):

    for arr in arrays[1:]:
        assert len(arrays[0]) == len(arr), "Arrays must be of the same length"
    
    fig, ax = plt.subplots()

    y_min = float('inf')
    y_max = -float('inf')
    for i, arr in enumerate(arrays):
        ax.plot(arr, label=f'Array {i}', color=get_random_color(seed+i))
        y_min_ = min(arr)
        y_max_ = max(arr)
        y_min = y_min_ if y_min_ < y_min else y_min
        y_max = y_max_ if y_max_ > y_max else y_max

    ax.set_ylim(y_min, y_max)


    # Add labels, legend, and title
    ax.set_xlabel('idx')
    ax.set_ylabel('val')
    ax.set_title('Two Arrays')
    ax.legend()

    # Show or save plot
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_heatmap(
        array, 
        title='Heatmap', 
        xlabel='X-axis', 
        ylabel='Y-axis', 
        colorbar_label='Values', 
        cmap='viridis',
        path=None
    ):
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.imshow(array, aspect='auto', cmap=cmap)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show or save plot
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_side_by_side_heatmaps(matrix1, matrix2, cmap='viridis'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    sns.heatmap(matrix1, ax=axes[0], cmap=cmap, cbar=True)
    axes[0].set_title('Heatmap of Matrix 1')


    sns.heatmap(matrix2, ax=axes[1], cmap=cmap, cbar=True)
    axes[1].set_title('Heatmap of Matrix 2')

    plt.tight_layout()
    plt.show()
    