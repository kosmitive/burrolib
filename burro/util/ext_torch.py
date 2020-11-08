import torch
import numpy as np


def tile(a, dim, n_tile):
    """TIle the input array along the dimension.

    :param a: The tensor to repeat.
    :param dim: The dimension to repeat.
    :param n_tile: The number of times the dimension should be repeated.
    :return: The tiles torch tensor.
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
