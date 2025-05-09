import torch
import numpy as np
import os


def get_abc_randn(n_points, ground_truth='H0', dim=2, device='cuda:0', **ignored):
    dim = 2
    c = torch.randn(n_points, dim, device=device)
    c /= torch.norm(c, dim=1, keepdim=True)

    noise = 0.05 * torch.randn(n_points, dim, device=device)
    b = c + noise
    a = c.clone()

    if ground_truth == 'H1':
        a[:, 0] += noise[:, 0]
        a[:, 1:] += 0.05 * torch.randn_like(a[:, 1:], device=device)
    elif ground_truth == 'H0':
        a += 0.05 * torch.randn_like(a)
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    return a, b, c


def load_rat_data(seed, n_points, ground_truth, dim, device):
    max_n_points = 3000
    path = os.path.join(os.environ['SCRATCH'],
                        f'ratinabox_saved_data/{dim}_cells_{max_n_points}_points_noise_0.1_seed_{seed}.npy')
    data = np.load(path, allow_pickle=True).item()

    # subindices
    rng = np.random.default_rng(seed=int(1e4) + seed)
    indices = rng.permutation(max_n_points)[:n_points]

    if ground_truth == 'H0':
        a = torch.relu(torch.tensor(data['head_dir_ind_rate'][indices], device=device, dtype=torch.float32))
    else:
        a = torch.relu(torch.tensor(data['head_dir_rate'][indices], device=device, dtype=torch.float32))

    b_1 = torch.relu(torch.tensor(data['head_dir_rate'][indices], device=device, dtype=torch.float32))
    b_2 = torch.tensor(data['grid_rate'][indices], device=device, dtype=torch.float32)
    b = torch.relu(b_1 + b_2 - 1)

    c_pos = torch.tensor(data['pos'][indices], device=device, dtype=torch.float32)
    c_hd = torch.tensor(data['head_direction'][indices], device=device, dtype=torch.float32)
    c = torch.cat((c_pos, c_hd), dim=1)

    return a, b, c
