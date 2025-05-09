import torch
import math
import numpy as np


def normalize_xy(x, y=None):
    if len(x.shape) != 2 or (y is not None and len(y.shape) != 2):
        raise ValueError(f'x/y should be 2-dim, but x is {len(x.shape)}-dim and y is {len(y.shape)}-dim')

    x = x / torch.linalg.norm(x, dim=1, keepdim=True)
    if y is not None:
        y = y / torch.linalg.norm(y, dim=1, keepdim=True)
    return x, y


def compute_pdist_sq(x, y=None):
    """compute the squared paired distance between x and y."""
    if len(x.shape) == 1:
        if y is None:
            y = x.clone()
        return (x[:, None] - y[None, :]) ** 2

    if len(x.shape) != 2:
        raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')
    if y is not None:
        if len(y.shape) != 2:
            raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')

        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
        y_norm = torch.linalg.norm(y, dim=1, keepdim=False)[None, :]

        return torch.clamp(x_norm ** 2 + y_norm ** 2 - 2.0 * x @ y.T, min=0)
    a = x.reshape(x.shape[0], -1)
    aTa = a @ a.T
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)


def linear_kernel(x, y=None, normalized=False, **ignored):
    if len(x.shape) != 2:
        raise ValueError(f'x should be 2-dim, but it is {len(x.shape)}-dim')
    if y is not None:
        if len(y.shape) != 2:
            raise ValueError(f'x should be 2-dim, but it is {len(x.shape)}-dim')

    if normalized:
        x, y = normalize_xy(x, y)
    if y is None:
        y = x
    return x @ y.T


def cossim_kernel(x, y=None, **ignored):
    return linear_kernel(x, y, normalized=True)


def gaussian_kernel(x, y=None, normalized=False, sigma2=1.0, **ignored):
    if normalized:
        x, y = normalize_xy(x, y)

    return torch.exp(-compute_pdist_sq(x, y) / sigma2)


def gaussian_precision_kernel(x, y=None, normalized=False, gamma=1.0, int_bin_mask=None, **ignored):
    if normalized:
        x, y = normalize_xy(x, y)

    if isinstance(gamma, float) or (isinstance(gamma, torch.Tensor) and np.prod(gamma.shape) == 1):
        gamma = gamma * torch.ones(x.shape[1], device=x.device)
    assert len(gamma.shape) == 1 and len(gamma) == x.shape[1], gamma

    if int_bin_mask is not None:
        str_bin_mask = bin(int(int_bin_mask))[2:].zfill(4)
        mask = torch.tensor(list(map(int, list(str_bin_mask))), dtype=bool, device=x.device)

        gamma = gamma / math.sqrt(mask.sum())
        gamma[~mask] = 0.0
    else:
        gamma = gamma / math.sqrt(len(gamma))

    return torch.exp(-compute_pdist_sq(x * gamma[None, :], None if y is None else y * gamma[None, :]))


def gaussian_precision_with_kronecker_kernel(x, y=None, normalized=False, gamma=1.0, int_bin_mask=None, **ignored):
    if normalized:
        x, y = normalize_xy(x, y)

    if isinstance(gamma, float) or (isinstance(gamma, torch.Tensor) and np.prod(gamma.shape) == 1):
        gamma = gamma * torch.ones(x.shape[1], device=x.device)
    assert len(gamma.shape) == 1 and len(gamma) == x.shape[1], gamma

    if int_bin_mask is not None:
        str_bin_mask = bin(int(int_bin_mask))[2:].zfill(len(gamma))
        mask = torch.tensor(list(map(int, list(str_bin_mask))), dtype=bool, device=x.device)

        gamma = gamma / math.sqrt(mask.sum())
        gamma[~mask] = 0.0
    else:
        gamma = gamma / math.sqrt(len(gamma))

    gaussian_part = torch.exp(-compute_pdist_sq(x * gamma[None, :], None if y is None else y * gamma[None, :]))
    kronecker_part = kronecker_kernel(x[:, gamma == 0], None if y is None else y[:, gamma == 0])
    return gaussian_part + kronecker_part


def imq_kernel(x, y=None, normalized=False, sigma2=1.0, **ignored):
    if normalized:
        x, y = normalize_xy(x, y)

    return 1 / torch.sqrt(1 + compute_pdist_sq(x, y) / sigma2)


def poly_kernel(x, y=None, n=1, c=1.0, normalized=False, **ignored):
    return (linear_kernel(x, y, normalized) + c) ** n


def poly_decaying_kernel(x, y=None, n=1, alpha=1.0, **ignored):
    return (alpha * linear_kernel(x, y, normalized=True) + 1.0) ** n / (1 + alpha) ** n


def kronecker_kernel(x, y=None, **ignored):
    if len(x.shape) != 2:
        raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')

    if y is None:
        y = x.clone()
    else:
        if len(y.shape) != 2:
            raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')
    return (x[:, None, :] == y[None, :, :]).all(dim=-1).float()
