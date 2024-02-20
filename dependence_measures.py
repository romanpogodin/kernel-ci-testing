from abc import ABC, abstractmethod
import torch
import cond_mean_estimation as cme
import numpy as np
from warnings import warn
import kernels
import pval_computations
from scipy.stats import norm as norm_distr


def compute_hsic(Kx, Ky, biased=True):
    n = Kx.shape[0]

    if biased:
        a_vec = Kx.mean(dim=0)
        b_vec = Ky.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

    else:
        tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
        tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

        u = tilde_Kx * tilde_Ky
        k_row = tilde_Kx.sum(dim=1)
        l_row = tilde_Ky.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_Kx.sum()
        mu_y = tilde_Ky.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean


class DependenceMeasure(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def find_regressors(self, x, z, y):
        pass

    @abstractmethod
    def compute_statistic(self, x, z, y):
        pass

    @abstractmethod
    def compute_pval(self, statistic_value, pval_approx_type):
        pass


class KCIMeasure(DependenceMeasure):
    def __init__(self, kernel_x, kernel_yx, kernel_z, kernel_yz,
                 kernel_yx_args, kernel_x_args, kernel_yz_args, kernel_z_args, biased=True):
        super().__init__()
        if not (kernels.is_kernel_universal(kernel_x) and kernels.is_kernel_universal(kernel_z) and (
                kernels.is_kernel_universal(kernel_yx) or kernels.is_kernel_universal(kernel_yz))):
            warn('Some of the kernels are not universal. '
                 'Both x/z, and at least one of yx/yz should be universal to guarantee asymptotic power=1.')

        self.register_buffer('K_yy_inv_z', torch.empty(1, dtype=torch.float))
        self.register_buffer('K_yy_inv_K_zz', torch.empty(1, dtype=torch.float))
        self.kernel_yz_args = kernel_yz_args
        self.kernel_z_args = kernel_z_args

        self.register_buffer('K_yy_inv_x', torch.empty(1, dtype=torch.float))
        self.register_buffer('K_yy_inv_K_xx', torch.empty(1, dtype=torch.float))
        self.kernel_yx_args = kernel_yx_args
        self.kernel_x_args = kernel_x_args

        self.kernel_x = kernel_x
        self.kernel_yx = kernel_yx
        self.kernel_z = kernel_z
        self.kernel_yz = kernel_yz

        self.register_buffer('x_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('y_x_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('z_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('y_z_holdout', torch.empty(1, dtype=torch.float))

        self.register_buffer('idx_splits_half_split_yz', torch.empty(1, dtype=torch.int))
        self.yz_args_list = None
        self.register_buffer('idx_splits_half_split_yx', torch.empty(1, dtype=torch.int))
        self.yx_args_list = None

        self._regression_done = False

        self.biased = biased

    def find_regressors(self, x, z, y_z, y_x=None, param_dict_yz=None, lambda_values_yz=None, param_dict_yx=None,
                        lambda_values_yx=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128,
                        half_split_yx_estimator=False, half_split_yz_estimator=False, **ignored):
        if y_x is None:
            y_x = y_z

        self.x_holdout = x.detach().clone()
        self.y_x_holdout = y_x.detach().clone()
        self.z_holdout = z.detach().clone()
        self.y_z_holdout = y_z.detach().clone()

        if half_split_yz_estimator:
            self.idx_splits_half_split_yz, self.K_yy_inv_z, self.yz_args_list = cme.get_yz_regressors_half_split(
                y_z, z,
                self.kernel_yz,
                self.kernel_z,
                self.kernel_yz_args,
                self.kernel_z_args,
                param_dict_yz,
                lambda_values_yz,
                verbose,
                cpu_solver,
                cpu_dtype
            )
        else:
            self.K_yy_inv_z, self.K_yy_inv_K_zz, self.kernel_yz_args = \
                cme.get_yz_regressors(y_z, z, self.kernel_yz, self.kernel_z, self.kernel_yz_args, self.kernel_z_args,
                                      param_dict_yz, lambda_values_yz, verbose, cpu_solver, cpu_dtype)
            if isinstance(self.kernel_yz, list):
                self.kernel_yz = self.kernel_yz_args[0]
                self.kernel_yz_args = self.kernel_yz_args[1]

        if half_split_yx_estimator:
            self.idx_splits_half_split_yx, self.K_yy_inv_x, self.yx_args_list = cme.get_yz_regressors_half_split(
                y_x, x,
                self.kernel_yx,
                self.kernel_x,
                self.kernel_yx_args,
                self.kernel_x_args,
                param_dict_yx,
                lambda_values_yx,
                verbose,
                cpu_solver,
                cpu_dtype
            )
        else:
            self.K_yy_inv_x, self.K_yy_inv_K_xx, self.kernel_yx_args = \
                cme.get_yz_regressors(y_x, x, self.kernel_yx, self.kernel_x, self.kernel_yx_args, self.kernel_x_args,
                                      param_dict_yx, lambda_values_yx, verbose, cpu_solver, cpu_dtype)
            if isinstance(self.kernel_yx, list):
                self.kernel_yx = self.kernel_yx_args[0]
                self.kernel_yx_args = self.kernel_yx_args[1]

        self._regression_done = True

    def compute_statistic(self, x, z, y, return_matrices=False):
        # testing x _|| z | y
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        if len(self.K_yy_inv_x.shape) == 3:
            K_xx_c = cme.center_matrix_with_cme_half_split(y, x, self.y_x_holdout, self.x_holdout,
                                                           self.idx_splits_half_split_yx, self.K_yy_inv_x, self.kernel_yx,
                                                           self.kernel_x, self.yx_args_list, self.kernel_x_args)
        else:
            K_xx_c = cme.center_matrix_with_cme(y, x, self.y_x_holdout, self.x_holdout,
                                                self.K_yy_inv_x, self.K_yy_inv_K_xx,
                                                self.kernel_yx, self.kernel_x,
                                                self.kernel_yx_args, self.kernel_x_args)
        if len(self.K_yy_inv_z.shape) == 3:
            K_zz_c = cme.center_matrix_with_cme_half_split(y, z, self.y_z_holdout, self.z_holdout,
                                                           self.idx_splits_half_split_yz, self.K_yy_inv_z,
                                                           self.kernel_yz,
                                                           self.kernel_z, self.yz_args_list, self.kernel_z_args)
        else:
            K_zz_c = cme.center_matrix_with_cme(y, z, self.y_z_holdout, self.z_holdout,
                                                self.K_yy_inv_z, self.K_yy_inv_K_zz,
                                                self.kernel_yz, self.kernel_z,
                                                self.kernel_yz_args, self.kernel_z_args)

        K_yy = eval(f'kernels.{self.kernel_yz}_kernel(y, **self.kernel_yz_args)')
        K_zz_c = K_zz_c * K_yy

        statistic_value = compute_hsic(K_xx_c, K_zz_c, self.biased)

        if return_matrices:
            return statistic_value, K_xx_c, K_zz_c
        return statistic_value

    def compute_pval(self, statistic_value, pval_approx_type, **kwargs):
        if not self.biased and pval_approx_type == 'gamma':
            raise NotImplementedError('P-value calculation only works for the biased statistic')

        if pval_approx_type == 'gamma':
            return pval_computations.compute_gamma_pval_approximation(
                statistic_value, K=kwargs['K'], L=kwargs['L'], return_params=False)
        elif pval_approx_type == 'wild':
            return pval_computations.compute_wild_bootstrap_pval(
                statistic_value, K=kwargs['K'], L=kwargs['L'],
                return_params=False, n_samples=kwargs['n_samples'] if 'n_samples' in kwargs else 1000,
                compute_stat_func=lambda x, y: compute_hsic(x, y, self.biased),
                chunk_size=kwargs['chunk_size'] if 'chunk_size' in kwargs else None)
        else:
            raise NotImplementedError(f'{pval_approx_type} pval_approx_type is not supported.')

    def compute_xsplit_resampling_snr(self, x, z, y, x_ho, z_ho, y_ho_z, y_ho_x=None, param_dict_yz=None,
                                      lambda_values_yz=None, param_dict_yx=None, lambda_values_yx=None,
                                      verbose=False, cpu_solver=False, cpu_dtype=np.float128,
                                      half_split_yx_estimator=False, n_resamples=30):
        if not half_split_yx_estimator:
            raise ValueError(f'half_split_yx_estimator={half_split_yx_estimator} has to be True.')

        resampled_kci = torch.zeros(n_resamples)

        for i in range(n_resamples):
            self.find_regressors(x_ho, z_ho, y_ho_z, y_ho_x, param_dict_yz, lambda_values_yz, param_dict_yx,
                                 lambda_values_yx, verbose, cpu_solver, cpu_dtype, half_split_yx_estimator)
            resampled_kci[i] = self.compute_statistic(x, z, y)
        return resampled_kci.mean() / resampled_kci.std()


class CirceMeasure(KCIMeasure):
    def __init__(self, kernel_x, kernel_z, kernel_yz, kernel_yz_args, kernel_z_args, kernel_x_args, biased=True):
        super().__init__(kernel_x, kernel_yx=None, kernel_z=kernel_z, kernel_yz=kernel_yz,
                         kernel_yx_args=None, kernel_x_args=kernel_x_args,
                         kernel_yz_args=kernel_yz_args, kernel_z_args=kernel_z_args, biased=biased,)

    def find_regressors(self, x, z, y_z, param_dict_yz=None, lambda_values_yz=None, verbose=False, cpu_solver=False,
                        cpu_dtype=np.float128, half_split_yz_estimator=False, **ignored):
        if half_split_yz_estimator:
            self.idx_splits_half_split_yz, self.K_yy_inv_z, self.yz_args_list = cme.get_yz_regressors_half_split(
                y_z, z,
                self.kernel_yz,
                self.kernel_z,
                self.kernel_yz_args,
                self.kernel_z_args,
                param_dict_yz,
                lambda_values_yz,
                verbose,
                cpu_solver,
                cpu_dtype
            )
        else:
            self.K_yy_inv_z, self.K_yy_inv_K_zz, self.kernel_yz_args = \
                cme.get_yz_regressors(y_z, z, self.kernel_yz, self.kernel_z, self.kernel_yz_args, self.kernel_z_args,
                                      param_dict_yz, lambda_values_yz, verbose, cpu_solver, cpu_dtype)
            if isinstance(self.kernel_yz, list):
                self.kernel_yz = self.kernel_yz_args[0]
                self.kernel_yz_args = self.kernel_yz_args[1]

        self.x_holdout = x.detach().clone()
        self.z_holdout = z.detach().clone()
        self.y_z_holdout = y_z.detach().clone()

        self._regression_done = True

    def compute_statistic(self, x, z, y, return_matrices=False):
        # testing x _|| z | y
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        K_xx = eval(f'kernels.{self.kernel_x}_kernel(x, **self.kernel_x_args)')
        if len(self.K_yy_inv_z.shape) == 3:
            K_zz_c = cme.center_matrix_with_cme_half_split(y, z, self.y_z_holdout, self.z_holdout,
                                                           self.idx_splits_half_split_yz, self.K_yy_inv_z,
                                                           self.kernel_yz,
                                                           self.kernel_z, self.yz_args_list, self.kernel_z_args)
        else:
            K_zz_c = cme.center_matrix_with_cme(y, z, self.y_z_holdout, self.z_holdout,
                                                self.K_yy_inv_z, self.K_yy_inv_K_zz,
                                                self.kernel_yz, self.kernel_z,
                                                self.kernel_yz_args, self.kernel_z_args)

        K_yy = eval(f'kernels.{self.kernel_yz}_kernel(y, **self.kernel_yz_args)')
        K_zz_c = K_zz_c * K_yy

        statistic_value = compute_hsic(K_xx, K_zz_c, self.biased)

        if return_matrices:
            return statistic_value, K_xx, K_zz_c
        return statistic_value

    def compute_xsplit_resampling_snr(self, **ignored):
        raise NotImplementedError('This method is only intended for KCI, not CIRCE.')


class GCMMeasure(DependenceMeasure):
    def __init__(self, kernel_yx, kernel_yz, kernel_yx_args, kernel_yz_args):
        super().__init__()

        self.register_buffer('K_yy_inv_Phi_z', torch.empty(1, dtype=torch.float))
        self.kernel_yz_args = kernel_yz_args
        self.kernel_z_args = {'n': 1}

        self.register_buffer('K_yy_inv_Phi_x', torch.empty(1, dtype=torch.float))
        self.kernel_yx_args = kernel_yx_args
        self.kernel_x_args = {'n': 1}

        self.kernel_x = 'linear'
        self.kernel_yx = kernel_yx
        self.kernel_z = 'linear'
        self.kernel_yz = kernel_yz

        self.register_buffer('x_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('y_x_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('z_holdout', torch.empty(1, dtype=torch.float))
        self.register_buffer('y_z_holdout', torch.empty(1, dtype=torch.float))

        self._regression_done = False

    def find_regressors(self, x, z, y_z, y_x=None, param_dict_yz=None, lambda_values_yz=None, param_dict_yx=None,
                        lambda_values_yx=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128, **ignored):
        if y_x is None:
            y_x = y_z

        self.x_holdout = x.detach().clone()
        self.y_x_holdout = y_x.detach().clone()
        self.z_holdout = z.detach().clone()
        self.y_z_holdout = y_z.detach().clone()

        K_yy_inv_z, _, self.kernel_yz_args = \
            cme.get_yz_regressors(y_z, z, self.kernel_yz, self.kernel_z, self.kernel_yz_args, self.kernel_z_args,
                                  param_dict_yz, lambda_values_yz, verbose, cpu_solver, cpu_dtype)

        self.K_yy_inv_Phi_z = K_yy_inv_z @ self.z_holdout

        if isinstance(self.kernel_yz, list):
            self.kernel_yz = self.kernel_yz_args[0]
            self.kernel_yz_args = self.kernel_yz_args[1]

        K_yy_inv_x, _, self.kernel_yx_args = \
            cme.get_yz_regressors(y_x, x, self.kernel_yx, self.kernel_x, self.kernel_yx_args, self.kernel_x_args,
                                  param_dict_yx, lambda_values_yx, verbose, cpu_solver, cpu_dtype)
        self.K_yy_inv_Phi_x = K_yy_inv_x @ self.x_holdout

        if isinstance(self.kernel_yx, list):
            self.kernel_yx = self.kernel_yx_args[0]
            self.kernel_yx_args = self.kernel_yx_args[1]

        self._regression_done = True

    def compute_statistic(self, x, z, y, **ignored):
        # testing x _|| z | y
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        K_Yy_x = eval(f'kernels.{self.kernel_yx}_kernel(self.y_x_holdout, y=y, **self.kernel_yx_args)')
        K_Yy_z = eval(f'kernels.{self.kernel_yz}_kernel(self.y_z_holdout, y=y, **self.kernel_yz_args)')

        residual_x = x - K_Yy_x.T @ self.K_yy_inv_Phi_x
        residual_z = z - K_Yy_z.T @ self.K_yy_inv_Phi_z

        r_vals = residual_x[:, :, None] * residual_z[:, None, :]

        n = r_vals.shape[0]
        gcm_mean = r_vals.mean(dim=0)
        tau_d = ((r_vals ** 2).mean(dim=0) - gcm_mean ** 2) ** 0.5

        statistic_value = np.sqrt(n) * gcm_mean / tau_d
        statistic_value = torch.abs(statistic_value).max()

        if x.shape[1] * z.shape[1] == 1:
            return statistic_value, None

        r_vals = r_vals.view((n, -1)).T
        gcm_mean = gcm_mean.view((-1, 1))
        sigma_half = (r_vals - gcm_mean) / np.sqrt(n) / tau_d.view((-1, 1))

        return statistic_value, sigma_half

    def compute_pval(self, statistic_value, sigma_half, **kwargs):
        return pval_computations.compute_gcm_pval(statistic_value, sigma_half,
                                                  n_samples=kwargs['n_samples'] if 'n_samples' in kwargs else 1000)


class RBPT2Measure(DependenceMeasure):
    def __init__(self, kernel_w, kernel_y, kernel_w_args, kernel_y_args):
        super().__init__()

        if kernel_w == 'linreg':
            self.g = cme.LinearRegression()
        else:
            self.g = cme.KernelRidgeRegression(kernel_w, 'linear', kernel_w_args, dict())
        self.h = cme.KernelRidgeRegression(kernel_y, 'linear', kernel_y_args, dict())

        self._regression_done = False

    def find_regressors(self, x, z, y_z, y_x=None, param_dict_rbpt_w=None, lambda_values_rbpt_w=None, 
                        param_dict_rbpt_y=None, lambda_values_rbpt_y=None, verbose=False, cpu_solver=False,
                        cpu_dtype=np.float128, z_x=None, **ignored):
        # testing x _||_ z | y, and z/y is unlabeled
        if y_x is None:
            y_x = y_z
            z_x = z
        if y_x is not None and z_x is None:
            raise ValueError('Unlabelled data should contain both y_x and z_x, but only y_x was given.')

        self.g.fit(torch.hstack((z_x, y_x)), x, verbose=verbose, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype,
                   param_dict_x=param_dict_rbpt_w, lambda_values_x=lambda_values_rbpt_w)
        g_predictions = self.g.predict(torch.hstack((z, y_z)))
        self.h.fit(y_z, g_predictions, verbose=verbose, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype,
                   param_dict_x=param_dict_rbpt_y, lambda_values_x=lambda_values_rbpt_y)

        self._regression_done = True

    def compute_statistic(self, x, z, y, **ignored):
        # testing x _||_ z | y, and z/y is unlabeled
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        n = x.shape[0]

        g_predictions = self.g.predict(torch.hstack((z, y)))
        h_predictions = self.h.predict(y)

        loss1 = ((x - g_predictions) ** 2).mean(dim=1)
        loss2 = ((x - h_predictions) ** 2).mean(dim=1)

        # not in the original paper, but improves p-values
        bias_correction = ((h_predictions - g_predictions) ** 2).mean(dim=1)

        statistic_value = loss2 - loss1 + bias_correction
        statistic_value = np.sqrt(n) * statistic_value.mean() / statistic_value.std()
        return statistic_value

    def compute_pval(self, statistic_value, **kwargs):
        return norm_distr.sf(statistic_value.item())
