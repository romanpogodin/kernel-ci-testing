import torch
from scipy.stats import gamma as gamma_distr
from scipy.stats import norm as norm_distr


def center_kernel_matrix(K):
    K = K - K.mean(axis=1, keepdims=True)
    return K - K.mean(axis=0, keepdims=True)


def compute_gamma_pval_approximation(statistic_value, K, L, return_params=False):
    # From KCI: https://arxiv.org/abs/1202.3775
    K = center_kernel_matrix(K)
    L = center_kernel_matrix(L)

    KL = K * L
    mean = torch.diagonal(KL).mean()

    var = 2 * (KL ** 2).mean()

    k = mean ** 2 / var
    theta = var / mean / K.shape[0]  # scaling fix wrt the paper

    if return_params:
        return gamma_distr.sf(statistic_value.item(), a=k.item(), loc=0, scale=theta.item()), k.item(), theta.item()
    return gamma_distr.sf(statistic_value.item(), a=k.item(), loc=0, scale=theta.item())


def compute_wild_bootstrap_pval(statistic_value, K, L, compute_stat_func, return_params=False, n_samples=1000,
                                chunk_size=None):
    Q = torch.randn((n_samples, K.shape[0]), device=K.device)[:, :, None]
    Q[Q >= 0] = 1
    Q[Q < 0] = -1

    def compute_single_val(rademacher_vals):
        KQ = (rademacher_vals * rademacher_vals.T) * K
        return compute_stat_func(KQ, L)

    compute_stat_vals = torch.vmap(compute_single_val, chunk_size=chunk_size)
    # todo: add an exception for OOM that suggests setting a chunk size
    stat_vals = compute_stat_vals(Q)

    pval = (stat_vals > statistic_value).float().mean().item()

    if return_params:
        return pval, stat_vals
    return pval


def compute_gcm_pval(statistic_value, sigma_half=None, n_samples=1000):
    if sigma_half is None or sigma_half.shape[0] == 1:
        return 2 * norm_distr.sf(statistic_value.item())

    bootstrap = torch.randn((sigma_half.shape[1], n_samples), device=sigma_half.device)
    bootstrap = torch.abs((sigma_half @ bootstrap)).max(dim=0).values

    return ((bootstrap >= statistic_value).sum() + 1) / (1 + n_samples)
