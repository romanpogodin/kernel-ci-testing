import numpy as np
import torch
from copy import deepcopy
import kernels
from scipy.linalg import solve as scp_solve
from warnings import warn


def add_diag(x, val):
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f'x is not a square matrix: shape {x.shape}')

    idx = range(x.shape[0])
    y = x.clone()
    y[idx, idx] += val
    return y


def compute_cme_error(K_YY, K_QQ, K_Yy, K_Qq, K_qq, reg):
    n = K_YY.shape[0]
    Kinv = torch.linalg.solve(add_diag(K_YY, n * reg), K_Yy).T

    cme_error = (K_qq.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() - 2 * (Kinv @ K_Qq).diagonal()).mean()

    return cme_error


def compute_single_k_fold_error(K_yy, K_QQ, reg, k=2):
    n = K_yy.shape[0]
    idx = torch.randperm(n, device=K_yy.device)

    k_fold_error = 0
    for i in range(k):
        idx_test = idx[i * (n // k):min(n, (i + 1) * (n // k))]
        idx_train = torch.tensor(np.setdiff1d(idx.cpu().numpy(), idx_test.cpu().numpy()), device=idx.device)
        k_fold_error += compute_cme_error(K_YY=K_yy[idx_train][:, idx_train], K_QQ=K_QQ[idx_train][:, idx_train],
                                          K_Yy=K_yy[idx_train][:, idx_test], K_Qq=K_QQ[idx_train][:, idx_test],
                                          K_qq=K_QQ[idx_test][:, idx_test], reg=reg)

    return k_fold_error / k


def compute_single_loo_error(K_yy, K_QQ, reg, cpu_solver=False, cpu_dtype=np.float128):
    n = K_yy.shape[0]
    if cpu_solver:
        A = cpu_dtype(add_diag(K_yy, n * reg).cpu().numpy())
        B = cpu_dtype(K_yy.cpu().numpy())
        Kinv = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_yy.device).T
    else:
        Kinv = torch.linalg.solve(add_diag(K_yy, n * reg), K_yy).T

    # without reg2, Kinv.T = Kinv
    return ((K_QQ.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() -
             2 * (Kinv @ K_QQ).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()


def compute_loo_errors(K_yy, K_QQ, lambda_values=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128):
    # Discard values below svd tolerance. Not multiplied by matrix size since it's done in compute_single_loo_error
    svd_tol = torch.linalg.matrix_norm(K_yy, ord=2) * torch.finfo(K_yy.dtype).eps

    if lambda_values is None:
        lambda_values = torch.logspace(1 + torch.log10(svd_tol), 7 + torch.log10(svd_tol), 7)
        lambda_values_tol = lambda_values
    else:
        lambda_values = torch.tensor(lambda_values, device=K_yy.device)
        lambda_values_tol = lambda_values[lambda_values >= svd_tol]

    if len(lambda_values_tol) == 0:
        raise ValueError(f'All lambda values < svd tolerance:\n{lambda_values} < {svd_tol}')

    loos = torch.zeros_like(lambda_values_tol)

    for i, value in enumerate(lambda_values_tol):
        loos[i] = compute_single_loo_error(K_yy, K_QQ, value, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    min_idx = torch.argmin(loos)

    lambda_vals = lambda_values_tol[min_idx]

    loos = loos.flatten()

    if verbose:
        print(f'lambda values: {lambda_values}\nsvd tolerance: {svd_tol}\nlambda > tol {lambda_values_tol}\n'
              f'LOOs: {loos}\nBest loo/loo: {loos[min_idx]}/{lambda_vals}')

    return loos[min_idx], lambda_vals


def leave_one_out_regressors_single_kernel(y, K_zz, kernel_y, lambda_values=None, param_dict=None, default_y_args=None,
                                           verbose=True, cpu_solver=False, cpu_dtype=np.float128):

    if param_dict is None:
        if verbose:
            print('No parameters to test for LOO found.'
                  ' LOO will be done with the passed/default ridge regression parameters')
            print(f'Kernel: {kernel_y}, default parameters: {default_y_args}')
        K_yy = eval(f'kernels.{kernel_y}_kernel(y, **default_y_args)')
        best_loo_error, best_loo_lambda = compute_loo_errors(K_yy, K_zz, lambda_values, verbose,
                                                             cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
        kernel_y_args = deepcopy(default_y_args)
    else:
        params_mesh = torch.meshgrid(*param_dict.values(), indexing='ij')
        param_names = param_dict.keys()
        loo_errors = torch.zeros_like(params_mesh[0].flatten())

        loo_lambda = torch.zeros_like(loo_errors)

        if verbose:
            print(f'Kernel: {kernel_y}, default parameters: {default_y_args}')

        for i in range(len(loo_errors)):
            kernel_y_args = deepcopy(default_y_args)
            for key_idx, key in enumerate(param_names):
                kernel_y_args[key] = params_mesh[key_idx].flatten()[i]

            K_yy = eval(f'kernels.{kernel_y}_kernel(y, **kernel_y_args)')
            loo_errors[i], best_loo_lambdas = compute_loo_errors(K_yy, K_zz, lambda_values, verbose,
                                                                 cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
            loo_lambda[i] = best_loo_lambdas

        min_idx = torch.argmin(loo_errors)

        kernel_y_args = deepcopy(default_y_args)
        for key_idx, key in enumerate(param_names):
            kernel_y_args[key] = params_mesh[key_idx].flatten()[min_idx]

        best_loo_error = loo_errors[min_idx]
        best_loo_lambda = loo_lambda[min_idx]

    if verbose:
        print(f'Best LOO: {best_loo_error}, Best parameters: lambda={best_loo_lambda} and {kernel_y_args}')

    return best_loo_lambda, kernel_y_args, best_loo_error


def leave_one_out_regressors(y, K_zz, kernel_y, lambda_values=None, param_dict=None, default_y_args=None, verbose=True,
                             cpu_solver=False, cpu_dtype=np.float128):
    if isinstance(kernel_y, list):
        best_loo_error = -1

        for kernel_y_name in kernel_y:
            loo_lambda, found_y_args, loo_error = leave_one_out_regressors_single_kernel(y, K_zz, kernel_y_name,
                                                                                         lambda_values,
                                                                                         param_dict[kernel_y_name],
                                                                                         default_y_args, verbose,
                                                                                         cpu_solver=cpu_solver,
                                                                                         cpu_dtype=cpu_dtype)

            if best_loo_error == -1 or best_loo_error > loo_error:
                best_loo_lambda = loo_lambda
                kernel_y_args = [kernel_y_name, found_y_args]
                best_loo_error = loo_error
        return best_loo_lambda, kernel_y_args, best_loo_error

    else:
        return leave_one_out_regressors_single_kernel(y, K_zz, kernel_y, lambda_values, param_dict, default_y_args,
                                                      verbose, cpu_solver=cpu_solver, cpu_dtype=np.float128)


def get_yz_regressors(y, z, kernel_y, kernel_z, kernel_y_args, kernel_z_args, param_dict=None, lambda_values=None,
                      verbose=True, cpu_solver=False, cpu_dtype=np.float128):
    n_points = y.shape[0]
    K_zz = eval(f'kernels.{kernel_z}_kernel(z, **kernel_z_args)')

    if verbose:
        print('Estimating regressions parameters with LOO')

    ridge_lambda, kernel_y_args, best_loo_error = leave_one_out_regressors(y, K_zz, kernel_y, lambda_values, param_dict,
                                                                           kernel_y_args, verbose,
                                                                           cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    if isinstance(kernel_y, list):
        K_yy = eval(f'kernels.{kernel_y_args[0]}_kernel(y, **kernel_y_args[1])')
    else:
        K_yy = eval(f'kernels.{kernel_y}_kernel(y, **kernel_y_args)')

    if verbose:
        print('All gram matrices computed')

    K_yy = add_diag(K_yy, K_yy.shape[0] * ridge_lambda)
    K_zz = torch.cat((torch.eye(n_points, device=K_yy.device), K_zz), 1)

    if cpu_solver:
        A = cpu_dtype(K_yy.cpu().numpy())
        B = cpu_dtype(K_zz.cpu().numpy())
        W_all = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_yy.device)
    else:
        W_all = torch.linalg.solve(K_yy, K_zz)

    K_yy_inv = W_all[:, :n_points]  # (K_yy + lambda n I)^(-1)
    K_yy_inv_K_zz = W_all[:, n_points:]  # (K_yy + lambda n I)^(-1) K_zz

    if verbose:
        print('W_all computed')

    return K_yy_inv, K_yy_inv_K_zz, kernel_y_args


def center_matrix_with_cme(y, x, y_ho, x_ho, K_yy_inv, K_yy_inv_K_xx,
                           kernel_y, kernel_x, kernel_y_args, kernel_x_args):
    K_xx_all = eval(f'kernels.{kernel_x}_kernel(torch.vstack((x, x_ho)), y=x, **kernel_x_args)')
    K_xx = K_xx_all[:x.shape[0]]
    K_Xx = K_xx_all[x.shape[0]:]

    K_Yy = eval(f'kernels.{kernel_y}_kernel(y_ho, y=y, **kernel_y_args)')

    A = (0.5 * K_Yy.T @ K_yy_inv_K_xx - K_Xx.T) @ K_yy_inv @ K_Yy

    return K_xx + A + A.T


def get_yz_regressors_half_split(y, z, kernel_y, kernel_z, kernel_y_args, kernel_z_args, param_dict=None,
                                 lambda_values=None, verbose=True, cpu_solver=False, cpu_dtype=np.float128):
    n_splits = 2
    n = y.shape[0]
    if n % n_splits != 0:
        # raise ValueError(f'n_poitns={n} should be divisible by n_splits={n_splits}')
        warn('n not divisible by 2, so the last point will be ignored.')
        y = y[:n_splits * (n // n_splits)]
        z = z[:n_splits * (n // n_splits)]
        n = n_splits * (n // n_splits)

    idx_splits = torch.randperm(n, device=y.device).view(n_splits, n // n_splits)

    y_args_list = list()
    K_yy_inv = torch.zeros((n_splits, n // n_splits, n // n_splits), device=y.device)

    for split in range(n_splits):
        Kinv, _, y_args = get_yz_regressors(y[idx_splits[split]], z[idx_splits[split]], kernel_y, kernel_z,
                                            kernel_y_args, kernel_z_args, param_dict, lambda_values, verbose,
                                            cpu_solver, cpu_dtype)
        K_yy_inv[split] = Kinv
        y_args_list.append(y_args)

    return idx_splits, K_yy_inv, y_args_list


def center_matrix_with_cme_half_split(y, x, y_ho, x_ho, idx_splits, K_yy_inv,
                                      kernel_y, kernel_x, kernel_y_args_list, kernel_x_args):
    if idx_splits.shape[0] != 2:
        raise NotImplementedError()
    n_splits = idx_splits.shape[0]
    n = y_ho.shape[0]
    if n % n_splits != 0:
        warn('n not divisible by 2, so the last point will be ignored.')
        y_ho = y_ho[:n_splits * (n // n_splits)]
        x_ho = x_ho[:n_splits * (n // n_splits)]
        n = n_splits * (n // n_splits)

    K_xx_all = eval(f'kernels.{kernel_x}_kernel(torch.vstack((x, x_ho)), **kernel_x_args)')
    K_xx = K_xx_all[:x.shape[0], :x.shape[0]]
    K_Xx = K_xx_all[x.shape[0] + idx_splits.flatten(), :x.shape[0]].view(idx_splits.shape[0], idx_splits.shape[1], -1)
    K_XX = K_xx_all[x.shape[0] + idx_splits.flatten()][:, x.shape[0] + idx_splits.flatten()].view(
        idx_splits.shape[0], idx_splits.shape[1], idx_splits.shape[0], idx_splits.shape[1])

    K_Yy = torch.zeros((idx_splits.shape[0], idx_splits.shape[1], y.shape[0]), device=y.device)

    for split in range(len(kernel_y_args_list)):
        if isinstance(kernel_y_args_list[split], list):
            kernel_y = kernel_y_args_list[split][0]
            kernel_y_args = kernel_y_args_list[split][1]
        else:
            kernel_y_args = kernel_y_args_list[split]
        K_Yy[split] = eval(f'kernels.{kernel_y}_kernel(y_ho[idx_splits[split]], y=y, **kernel_y_args)')

    # K_Xx  K_yy_inv K_Yy
    # [q, m, n] [q, m, m] [q, m, n]
    # A
    # [q, n, n]
    K_yy_inv_K_Yy = torch.bmm(K_yy_inv, K_Yy)
    A = -torch.bmm(K_Xx.transpose(1, 2), K_yy_inv_K_Yy)
    A = A + A.transpose(1, 2)
    # A, median_vals = torch.median(A, dim=0)

    # # K_yy_inv_K_Yy K_XX K_yy_inv_K_Yy
    # # [q, m, n] [q, m, q, m] [q, m, n]
    # # B
    # # [q, q, n, n]
    # # [qm, q, m, n]
    # B = torch.matmul(K_XX.transpose(1, 2), K_yy_inv_K_Yy[None, :, :, :])
    # # [q, q, n, n]
    # B = torch.matmul(B.transpose(-1, -2), K_yy_inv_K_Yy[:, None, :, :])

    # KxxT 2 2 n n, then take off-diagonals and corresponding Kyy vectors (01 with 1 and 10 with 0)
    B = torch.matmul(K_XX.transpose(1, 2).flatten(end_dim=1)[1:3], K_yy_inv_K_Yy.flip(dims=(0,)))
    B = torch.matmul(B.transpose(-1, -2), K_yy_inv_K_Yy)

    A = K_xx + 0.5 * (A.sum(dim=0) + B.sum(dim=0))

    return A


class KernelRidgeRegression(torch.nn.Module):
    def __init__(self, kernel_x, kernel_y, kernel_x_args, kernel_y_args):
        super().__init__()
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.kernel_x_args = kernel_x_args
        self.kernel_y_args = kernel_y_args

        self.register_buffer('x_train', torch.empty(1, dtype=torch.float))
        self.register_buffer('y_train', torch.empty(1, dtype=torch.float))
        self.register_buffer('Kxx_inv', torch.empty(1, dtype=torch.float))
        self.register_buffer('Kxx_inv_Kyy', torch.empty(1, dtype=torch.float))
        self.register_buffer('Kxx_inv_Y', torch.empty(1, dtype=torch.float))

        self._regression_done = False

    def fit(self, x, y, verbose=False, cpu_solver=True, cpu_dtype=np.float128,
            param_dict_x=None, lambda_values_x=None, **ignored):
        # predict y from x
        self.x_train = x.clone()
        self.y_train = y.clone()
        self.Kxx_inv, self.Kxx_inv_Kyy, self.kernel_x_args = \
            get_yz_regressors(self.x_train, self.y_train, self.kernel_x, self.kernel_y, self.kernel_x_args,
                              self.kernel_y_args, param_dict_x, lambda_values_x, verbose, cpu_solver, cpu_dtype)

        if isinstance(self.kernel_x, list):
            self.kernel_x = self.kernel_x_args[0]
            self.kernel_x_args = self.kernel_x_args[1]

        if self.kernel_y == 'linear':
            self.Kxx_inv_Y = self.Kxx_inv @ self.y_train

        self._regression_done = True

    def predict(self, x):
        if not self._regression_done:
            raise ValueError('Run .fit(x, y) first.')
        if self.kernel_y != 'linear':
            raise ValueError(f'Only linear y-kernel supports predict(); kernel_y={self.kernel_y}.')

        K_xX = eval(f'kernels.{self.kernel_x}_kernel(x, y=self.x_train, **self.kernel_x_args)')

        return K_xX @ self.Kxx_inv_Y

    def predict_kernel_matrix(self, x, y):
        return center_matrix_with_cme(x, y, self.x_train, self.y_train, self.Kxx_inv, self.Kxx_inv_Kyy,
                                      self.kernel_x, self.kernel_y, self.kernel_x_args, self.kernel_y_args)


class LinearRegression(torch.nn.Module):
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.register_buffer('weights', torch.empty(1, dtype=torch.float))
        self.fit_intercept = fit_intercept

    def fit(self, x, y, **ignored):
        if self.fit_intercept:
            x = torch.hstack((x, torch.ones((x.shape[0], 1), device=x.device)))
        self.weights = torch.linalg.lstsq(x, y).solution

    def predict(self, x):
        if self.fit_intercept:
            x = torch.hstack((x, torch.ones((x.shape[0], 1), device=x.device)))
        return x @ self.weights
