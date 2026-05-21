import torch

import ntops
from ntops.torch.utils import _cached_make


def roll(input, shifts, dims=None):
    if dims is None:
        flat = input.flatten()
        flat_out = torch.empty_like(flat)
        _roll_along_dim(flat, flat_out, shifts if isinstance(shifts, int) else shifts[0], 0)
        return flat_out.view(input.shape)

    dims_list = [dims] if isinstance(dims, int) else list(dims)
    shifts_list = [shifts] if isinstance(shifts, int) else list(shifts)

    current = input
    for shift, dim in zip(shifts_list, dims_list):
        out = torch.empty_like(current)
        _roll_along_dim(current, out, shift, dim)
        current = out

    return current


def _roll_along_dim(input, output, shift, dim):
    N = input.shape[dim]
    s = shift % N  # normalized non-negative shift

    kernel = _cached_make(ntops.kernels.roll.premake, input.ndim)

    if s == 0:
        kernel(input, output)
        return

    ndim = input.ndim

    def _idx(dim_slice):
        idx = [slice(None)] * ndim
        idx[dim] = dim_slice
        return tuple(idx)

    # output[..., :s, ...] = input[..., N-s:, ...]
    kernel(input[_idx(slice(N - s, N))], output[_idx(slice(0, s))])
    # output[..., s:, ...] = input[..., :N-s, ...]
    kernel(input[_idx(slice(0, N - s))], output[_idx(slice(s, N))])
