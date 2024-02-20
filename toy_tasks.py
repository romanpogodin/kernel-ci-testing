import torch
import numpy as np


__all__ = [
    'get_xzy_randn', 'get_xzy_circ', 'get_xzy_rbpt', 'get_xzy_randn_nl'
]


def get_xzy_randn(n_points, ground_truth='H0', dim=2, device='cuda:0', **ignored):
    y = torch.randn(n_points, dim, device=device)
    y /= torch.norm(y, dim=1, keepdim=True)

    noise = 0.1 * torch.randn(n_points, dim, device=device) / np.sqrt(dim)
    z = y + noise
    x = y.clone()

    if ground_truth == 'H1':
        x[:, 0] += noise[:, 0]
        x[:, 1:] += 0.1 * torch.randn_like(x[:, 1:], device=device) / np.sqrt(dim)
    elif ground_truth == 'H0':
        x += 0.1 * torch.randn_like(x) / np.sqrt(dim)
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    return x, z, y


def get_xzy_circ(n_points, ground_truth='H0', dim=2, device='cuda:0', **ignored):
    y = torch.randn(n_points, dim, device=device)
    y /= torch.norm(y, dim=1, keepdim=True)

    noise = 0.1 * torch.randn(n_points, dim, device=device) / np.sqrt(dim)
    z = y + noise

    if ground_truth == 'H1':
        x = y.clone()
        x[:, 0] += noise[:, 0]
        x[:, 1:] = 0.1 * torch.randn_like(x[:, 1:]) / np.sqrt(dim)
    elif ground_truth == 'H0':
        noise2 = 0.1 * torch.randn(n_points, dim, device=device) / np.sqrt(dim)
        x = y + noise2
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    y[y[:, 0] > 0] = 2 * y[y[:, 0] > 0]
    y[y[:, 1] < 0] = 0.5 * y[y[:, 1] < 0]

    return x, z, y


def get_xzy_rbpt(n_points, ground_truth='H0', dim=40, device='cuda:0', c=0.1, gamma=0.01, seed=1, **ignored):
    param_generator = torch.Generator(device=device)
    param_generator.manual_seed(seed)

    a = torch.randn(size=(dim, 1), generator=param_generator, device=device)
    b = torch.randn(size=(dim, 1), generator=param_generator, device=device)

    y = torch.randn(size=(n_points, dim), device=device)
    z = torch.randn(size=(n_points, 1), device=device) + (y @ b) ** 2
    x = torch.randn(size=(n_points, 1), device=device) + gamma * (y @ b) ** 2 + y @ a

    if ground_truth == 'H1':
         x += c * z
    elif ground_truth == 'H0':
        pass
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    return x / np.sqrt(dim), z / dim, y / np.sqrt(dim)


def get_xzy_randn_nl(n_points, ground_truth='H0', dim=2, device='cuda:0', **ignored):
    y = torch.randn(n_points, dim, device=device) / np.sqrt(dim)

    noise = 0.1 * torch.randn(n_points, dim, device=device) / np.sqrt(dim)
    z = y + noise
    x = y.clone()

    if ground_truth == 'H1':
        x[:, 0] += noise[:, 0]
        x[:, 1:] += 0.1 * torch.randn_like(x[:, 1:], device=device) / np.sqrt(dim)
    elif ground_truth == 'H0':
        x += 0.1 * torch.randn_like(x) / np.sqrt(dim)
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    x = x ** 2

    return x, z, y