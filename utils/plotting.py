import seaborn as sns
import matplotlib.pyplot as plt


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
