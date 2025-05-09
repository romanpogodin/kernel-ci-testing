import numpy as np
import torch
from copy import deepcopy
from scipy.linalg import solve as scp_solve
from warnings import warn
from splitkci import kernels


def add_diag(x, val):
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f'x is not a square matrix: shape {x.shape}')

    idx = range(x.shape[0])
    y = x.clone()
    y[idx, idx] += val
    return y


def compute_single_loo_error(K_xx, K_yy, reg, cpu_solver=False, cpu_dtype=np.float128):
    n = K_xx.shape[0]
    if cpu_solver:
        A = cpu_dtype(add_diag(K_xx, n * reg).cpu().numpy())
        B = cpu_dtype(K_xx.cpu().numpy())
        Kinv = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_xx.device).T
    else:
        Kinv = torch.linalg.solve(add_diag(K_xx, n * reg), K_xx).T

    # without reg2, Kinv.T = Kinv
    return ((K_yy.diagonal() + (Kinv @ K_yy @ Kinv.T).diagonal() -
             2 * (Kinv @ K_yy).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()


def compute_loo_errors(K_xx, K_yy, lambda_values=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128):
    # Discard values below svd tolerance. Not multiplied by matrix size since it's done in compute_single_loo_error
    svd_tol = torch.linalg.matrix_norm(K_xx, ord=2) * torch.finfo(K_xx.dtype).eps

    if lambda_values is None:
        lambda_values = torch.logspace(1 + torch.log10(svd_tol), 7 + torch.log10(svd_tol), 7)
        lambda_values_tol = lambda_values
    else:
        lambda_values = torch.tensor(lambda_values, device=K_xx.device)
        lambda_values_tol = lambda_values[lambda_values >= svd_tol]

    if len(lambda_values_tol) == 0:
        raise ValueError(f'All lambda values < svd tolerance:\n{lambda_values} < {svd_tol}')

    loos = torch.zeros_like(lambda_values_tol)

    for i, value in enumerate(lambda_values_tol):
        loos[i] = compute_single_loo_error(K_xx, K_yy, value, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    min_idx = torch.argmin(loos)

    lambda_vals = lambda_values_tol[min_idx]

    loos = loos.flatten()

    if verbose:
        print(f'lambda values: {lambda_values}\nsvd tolerance: {svd_tol}\nlambda > tol {lambda_values_tol}\n'
              f'LOOs: {loos}\nBest loo/loo: {loos[min_idx]}/{lambda_vals}')

    return loos[min_idx], lambda_vals


def leave_one_out_regressors_single_kernel(x, K_yy, kernel_x, lambda_values=None, param_dict=None,
                                           default_x_args=None, verbose=True, cpu_solver=False, cpu_dtype=np.float128):
    if param_dict is None:
        if verbose:
            print('No parameters to test for LOO found.'
                  ' LOO will be done with the passed/default ridge regression parameters')
            print(f'Kernel: {kernel_x}, default parameters: {default_x_args}')
        K_xx = eval(f'kernels.{kernel_x}_kernel(x, **default_x_args)')
        best_loo_error, best_loo_lambda = compute_loo_errors(K_xx, K_yy, lambda_values, verbose,
                                                             cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
        kernel_x_args = deepcopy(default_x_args)
    else:
        params_mesh = torch.meshgrid(*param_dict.values(), indexing='ij')
        param_names = param_dict.keys()
        loo_errors = torch.zeros_like(params_mesh[0].flatten())

        loo_lambda = torch.zeros_like(loo_errors)

        if verbose:
            print(f'Kernel: {kernel_x}, default parameters: {default_x_args}')

        for i in range(len(loo_errors)):
            kernel_x_args = deepcopy(default_x_args)
            for key_idx, key in enumerate(param_names):
                kernel_x_args[key] = params_mesh[key_idx].flatten()[i]

            K_xx = eval(f'kernels.{kernel_x}_kernel(x, **kernel_x_args)')
            loo_errors[i], best_loo_lambdas = compute_loo_errors(K_xx, K_yy, lambda_values, verbose,
                                                                 cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
            loo_lambda[i] = best_loo_lambdas

        min_idx = torch.argmin(loo_errors)

        kernel_x_args = deepcopy(default_x_args)
        for key_idx, key in enumerate(param_names):
            kernel_x_args[key] = params_mesh[key_idx].flatten()[min_idx]

        best_loo_error = loo_errors[min_idx]
        best_loo_lambda = loo_lambda[min_idx]

    if verbose:
        print(f'Best LOO: {best_loo_error}, Best parameters: lambda={best_loo_lambda} and {kernel_x_args}')

    return best_loo_lambda, kernel_x_args, best_loo_error


def leave_one_out_regressors(x, K_yy, kernel_x, lambda_values=None, param_dict=None, default_x_args=None, verbose=True,
                             cpu_solver=False, cpu_dtype=np.float128):
    if isinstance(kernel_x, list):
        best_loo_error = -1

        for kernel_x_name in kernel_x:
            loo_lambda, found_y_args, loo_error = leave_one_out_regressors_single_kernel(x, K_yy, kernel_x_name,
                                                                                         lambda_values,
                                                                                         param_dict[kernel_x_name],
                                                                                         default_x_args, verbose,
                                                                                         cpu_solver=cpu_solver,
                                                                                         cpu_dtype=cpu_dtype)

            if best_loo_error == -1 or best_loo_error > loo_error:
                best_loo_lambda = loo_lambda
                kernel_x_args = [kernel_x_name, found_y_args]
                best_loo_error = loo_error
        return best_loo_lambda, kernel_x_args, best_loo_error

    else:
        return leave_one_out_regressors_single_kernel(x, K_yy, kernel_x, lambda_values, param_dict, default_x_args,
                                                      verbose, cpu_solver=cpu_solver, cpu_dtype=np.float128)


def get_xy_regressors(x, y, kernel_x, kernel_y, kernel_x_args, kernel_y_args, param_dict=None, lambda_values=None,
                      verbose=True, cpu_solver=False, cpu_dtype=np.float128):
    # X to Y regression
    n_points = x.shape[0]
    K_yy = eval(f'kernels.{kernel_y}_kernel(y, **kernel_y_args)')

    if verbose:
        print('Estimating regressions parameters with LOO')

    ridge_lambda, kernel_x_args, best_loo_error = leave_one_out_regressors(
        x, K_yy, kernel_x, lambda_values, param_dict, kernel_x_args,
        verbose, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    if isinstance(kernel_x, list):
        K_xx = eval(f'kernels.{kernel_x_args[0]}_kernel(x, **kernel_x_args[1])')
    else:
        K_xx = eval(f'kernels.{kernel_x}_kernel(x, **kernel_x_args)')

    if verbose:
        print('All gram matrices computed')

    K_xx = add_diag(K_xx, K_xx.shape[0] * ridge_lambda)
    K_yy = torch.cat((torch.eye(n_points, device=K_xx.device), K_yy), 1)

    if cpu_solver:
        A = cpu_dtype(K_xx.cpu().numpy())
        B = cpu_dtype(K_yy.cpu().numpy())
        W_all = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_xx.device)
    else:
        W_all = torch.linalg.solve(K_xx, K_yy)

    K_xx_inv = W_all[:, :n_points]  # (K_xx + lambda n I)^(-1)
    K_xx_inv_K_yy = W_all[:, n_points:]  # (K_xx + lambda n I)^(-1) K_yy

    if verbose:
        print('W_all computed')

    return K_xx_inv, K_xx_inv_K_yy, kernel_x_args, ridge_lambda, best_loo_error


def center_matrix_with_cme(x, y, x_train, y_train,
                           K_xx_inv, K_xx_inv_K_yy,
                           kernel_x, kernel_y, kernel_x_args, kernel_y_args):
    K_yy_all = eval(f'kernels.{kernel_y}_kernel(torch.vstack((y, y_train)), y=y, **kernel_y_args)')
    K_yy = K_yy_all[:y.shape[0]]
    K_Yy = K_yy_all[y.shape[0]:]

    K_Xx = eval(f'kernels.{kernel_x}_kernel(x_train, y=x, **kernel_x_args)')

    A = (0.5 * K_Xx.T @ K_xx_inv_K_yy - K_Yy.T) @ K_xx_inv @ K_Xx

    return K_yy + A + A.T


class KernelRidgeRegression(torch.nn.Module):
    def __init__(self, kernel_x, kernel_y, kernel_x_args, kernel_y_args):
        super().__init__()
        self.kernel_x = deepcopy(kernel_x)
        self.kernel_y = deepcopy(kernel_y)
        self.kernel_x_args = deepcopy(kernel_x_args)
        self.kernel_y_args = deepcopy(kernel_y_args)

        self.ridge_lambda = 0
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
        self.Kxx_inv, self.Kxx_inv_Kyy, self.kernel_x_args, self.ridge_lambda, best_loo_error = \
            get_xy_regressors(self.x_train, self.y_train, self.kernel_x, self.kernel_y, self.kernel_x_args,
                              self.kernel_y_args, param_dict_x, lambda_values_x, verbose, cpu_solver, cpu_dtype)

        if isinstance(self.kernel_x, list):
            self.kernel_x = self.kernel_x_args[0]
            self.kernel_x_args = self.kernel_x_args[1]

        if self.kernel_y == 'linear':
            self.Kxx_inv_Y = self.Kxx_inv @ self.y_train

        self._regression_done = True
        return best_loo_error

    def eval_K_x_train_x(self, x):
        return eval(f'kernels.{self.kernel_x}_kernel(self.x_train, x, **self.kernel_x_args)')

    def eval_K_xx(self, x):
        return eval(f'kernels.{self.kernel_x}_kernel(x, **self.kernel_x_args)')

    def eval_K_y_train_y(self, y):
        return eval(f'kernels.{self.kernel_y}_kernel(self.y_train, y, **self.kernel_y_args)')

    def eval_K_yy(self, y):
        return eval(f'kernels.{self.kernel_y}_kernel(y, **self.kernel_y_args)')

    def predict(self, x):
        if not self._regression_done:
            raise ValueError('Run .fit(x, y) first.')
        if self.kernel_y != 'linear':
            raise ValueError(f'Only linear y-kernel supports predict(); kernel_y={self.kernel_y}.')

        K_xX = self.eval_K_x_train_x(x).T

        return K_xX @ self.Kxx_inv_Y

    def predict_kernel_matrix(self, x, y):
        if not self._regression_done:
            raise ValueError('Run .fit(x, y) first.')
        return center_matrix_with_cme(x, y, self.x_train, self.y_train, self.Kxx_inv, self.Kxx_inv_Kyy,
                                      self.kernel_x, self.kernel_y, self.kernel_x_args, self.kernel_y_args)

    def compute_cme_error(self, x, y):
        if not self._regression_done:
            raise ValueError('Run .fit(x, y) first.')
        return self.predict_kernel_matrix(x, y).diagonal().mean()


class SplitKernelRidgeRegression(torch.nn.Module):
    def __init__(self, kernel_x, kernel_y, kernel_x_args, kernel_y_args):
        super().__init__()
        self.krr_one = KernelRidgeRegression(kernel_x, kernel_y, kernel_x_args, kernel_y_args)
        self.krr_two = KernelRidgeRegression(kernel_x, kernel_y, kernel_x_args, kernel_y_args)

        self.register_buffer('idx_split_one', torch.empty(1, dtype=torch.float))
        self.register_buffer('idx_split_two', torch.empty(1, dtype=torch.float))
        self.register_buffer('Kx1_Ky12_Kx2', torch.empty(1, dtype=torch.float))

        self._regression_done = False

    def fit(self, x, y, verbose=False, cpu_solver=True, cpu_dtype=np.float128,
            param_dict_x=None, lambda_values_x=None, **ignored):
        idx_splits = torch.randperm(x.shape[0], device=x.device)
        if x.shape[0] % 2 == 1:
            warn('n not divisible by 2, so the last point will be ignored.')
        self.idx_split_one = idx_splits[:x.shape[0] // 2]
        self.idx_split_two = idx_splits[x.shape[0] // 2:2 * (x.shape[0] // 2)]

        best_loo_error_one = self.krr_one.fit(x[self.idx_split_one], y[self.idx_split_one],
                                              verbose, cpu_solver, cpu_dtype, param_dict_x, lambda_values_x)
        best_loo_error_two = self.krr_two.fit(x[self.idx_split_two], y[self.idx_split_two],
                                              verbose, cpu_solver, cpu_dtype, param_dict_x, lambda_values_x)
        Ky_one_y_two = eval(f'kernels.{self.krr_one.kernel_y}_kernel(y[self.idx_split_one], '
                            f'y[self.idx_split_two], **self.krr_one.kernel_y_args)')

        self.Kx1_Ky12_Kx2 = self.krr_one.Kxx_inv @ Ky_one_y_two @ self.krr_two.Kxx_inv
        self._regression_done = True
        return best_loo_error_one, best_loo_error_two

    def predict(self, x):
        raise NotImplementedError('predict only works for regular KRR with a linear kernel')

    def predict_kernel_matrix(self, x, y):
        K_Xone_x = self.krr_one.eval_K_x_train_x(x)
        K_Xtwo_x = self.krr_two.eval_K_x_train_x(x)
        K_Yone_y = self.krr_one.eval_K_y_train_y(y)
        K_Ytwo_y = self.krr_two.eval_K_y_train_y(y)

        K_xx_inv_K_Xx_one = self.krr_one.Kxx_inv @ K_Xone_x
        K_xx_inv_K_Xx_two = self.krr_two.Kxx_inv @ K_Xtwo_x
        A = -K_Yone_y.T @ K_xx_inv_K_Xx_one - K_Ytwo_y.T @ K_xx_inv_K_Xx_two
        B = K_Xone_x.T @ self.Kx1_Ky12_Kx2 @ K_Xtwo_x
        A = A + B
        K_yy = self.krr_one.eval_K_yy(y)
        return (A + A.T) / 2 + K_yy

    def compute_cme_error(self, x, y):
        return self.krr_one.predict_kernel_matrix(x, y).diagonal().mean(), \
            self.krr_two.predict_kernel_matrix(x, y).diagonal().mean()

    def eval_K_xx(self, x):
        return eval(f'kernels.{self.krr_one.kernel_x}_kernel(x, **self.krr_one.kernel_x_args)')


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


class NoRegression(torch.nn.Module):
    def __init__(self, kernel_x, kernel_y, kernel_x_args, kernel_y_args):
        super().__init__()
        self.kernel_x = deepcopy(kernel_x)
        self.kernel_y = deepcopy(kernel_y)
        self.kernel_x_args = deepcopy(kernel_x_args)
        self.kernel_y_args = deepcopy(kernel_y_args)

        self._regression_done = True

    def fit(self, *ignored, **ignored_too):
        return 0

    def eval_K_yy(self, y):
        return eval(f'kernels.{self.kernel_y}_kernel(y, **self.kernel_y_args)')

    def predict_kernel_matrix(self, x, y):
        return self.eval_K_yy(y)
