import seaborn as sns
import matplotlib.pyplot as plt


def plot_line_for_1d_loads(tensor, k):
    vals = tensor[k].numpy()
    sns.lineplot(x=range(len(vals)), y=vals)
    plt.title(f"Line Plot of tensor[{k}]")
    plt.xlabel("x")
    plt.ylabel("vals")
    plt.show()

def plot_heatmap_for_2d_loads(tensor, k):
    slice_tensor = tensor[k, :, :]
    sns.heatmap(slice_tensor.numpy(), cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Heatmap of tensor[{k}, :, :]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_heatmap_for_3d_loads(tensor, k, z):
    slice_tensor = tensor[k, :, :, z]
    sns.heatmap(slice_tensor.numpy(), cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Heatmap of tensor[{k}, :, :, {z}]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



# import os
# import torch


# dir_dataset = os.path.join('datasets', 'test_10-10')
# files = os.listdir(dir_dataset)
# data = []
# for f in files:
#     path = os.path.join(dir_dataset, f)
#     data_ = torch.load(path)
#     data.append(data_)
# data = torch.cat(data, dim=0)
# print(data.shape)

# for i in range(100, 110):
#     plot_heatmap_for_2d_loads(data, i)




# from simulate_data import (
#     BumpFunction2D,
#     BumpScheme2D1,
#     CornerPairLoading2D1,
#     CornerPairLoading2D2,
#     build_loadings,
#     TrigScheme1D1
# )

# grid_shape = [30]

# points = []
# indices = []
# for sz in grid_shape:
#     indices_ = torch.arange(sz, dtype=torch.int32)
#     points_ = indices_.to(torch.float64) / sz
#     indices.append(indices_)
#     points.append(points_)
# indices = torch.cartesian_prod(*indices)  # n-by-ndim
# points = torch.cartesian_prod(*points)    # n-by-ndim

# load_scheme = TrigScheme1D1()
# vals = load_scheme(points)
# loads = build_loadings(indices, grid_shape, vals)

# plot_line_for_1d_loads(loads, 0)
# plot_line_for_1d_loads(loads, 1)

# NOTE: Something is off with the loadings... checking BumpFunction2D

# bump_fcn = BumpFunction2D([0.25, 0.75], 0, [0.15, 0.15], 1)
# vals = bump_fcn(points)
# loads = torch.zeros(grid_shape, dtype=torch.float64)
# loads[indices[:,0], indices[:,1]] = vals

# # NOTE: BumpFunction2d behaves as expected...
# # NOTE: Something upstream of BumpFunction is behaving unexpectedly... testing CornerPair2D1

# load_fcn = CornerPair2D1()
# vals = load_fcn(points)
# loads = torch.zeros(grid_shape, dtype=torch.float64)
# loads[indices[:,0], indices[:,1]] = vals

# plot_heatmap_for_2d_loads(loads, 1)



# ---------- 3D ---------- #

# dataset = 'test_10-10-10'
# dir_dataset = os.path.join('datasets', dataset)
# path = os.path.join(dir_dataset, 'data-0.pt')
# data = torch.load(path)

# n = 25
# z = 2
# for i in range(n):
#     plot_heatmap_for_3d_loads(data, i, z)

# ---------- 2D ---------- #

# dataset = 'test_10-10'
# dir_dataset = os.path.join('datasets', dataset)
# path = os.path.join(dir_dataset, 'data-0.pt')
# data = torch.load(path)

# n = 25
# for i in range(n):
#     plot_heatmap_for_2d_loads(data, i)