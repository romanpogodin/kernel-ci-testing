from abc import ABC, abstractmethod
import torch
import numpy as np
from splitkci import cond_mean_estimation as cme
from splitkci import pval_computations
from scipy.stats import norm as norm_distr


def split_data(a_all, b_all, c_all, test_size_ratio=None, shuffle=True, n_test_points=None):
    if n_test_points is None:
        n_test_points = int(a_all.shape[0] * test_size_ratio)

    if shuffle:
        idx_all = torch.randperm(a_all.shape[0], device=a_all.device)
    else:
        idx_all = torch.arange(a_all.shape[0], device=a_all.device)

    idx_test, idx_train = idx_all[:n_test_points], idx_all[n_test_points:]

    a_test, b_test, c_test = a_all[idx_test], b_all[idx_test], c_all[idx_test]
    a_train, b_train, c_train = a_all[idx_train], b_all[idx_train], c_all[idx_train]

    return a_test, b_test, c_test, a_train, b_train, c_train


def compute_hsic(K, L, biased=True):
    n = K.shape[0]

    if biased:
        a_vec = K.mean(dim=0)
        b_vec = L.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return (K * L).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

    else:
        tilde_K = K - torch.diagflat(torch.diag(K))
        tilde_L = L - torch.diagflat(torch.diag(L))

        u = tilde_K * tilde_L
        k_row = tilde_K.sum(dim=1)
        l_row = tilde_L.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_K.sum()
        mu_y = tilde_L.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean


class DependenceMeasure(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def find_regressors(self, a, b, c_b, c_a=None, param_dict_cb=None, lambda_values_cb=None, param_dict_ca=None,
                        lambda_values_ca=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128, **ignored):
        if c_a is None:
            c_a = c_b

        loo_ca = self.krr_ca.fit(c_a, a, verbose, cpu_solver, cpu_dtype, param_dict_ca, lambda_values_ca)
        loo_cb = self.krr_cb.fit(c_b, b, verbose, cpu_solver, cpu_dtype, param_dict_cb, lambda_values_cb)

        self._regression_done = True
        return loo_ca, loo_cb

    @abstractmethod
    def compute_statistic(self, a, b, c):
        pass

    @abstractmethod
    def compute_pval(self, statistic_value, pval_approx_type):
        pass

    def _split_data(self, a, b, c, a_aux=None, b_aux=None, c_aux=None, train_test_split=False):
        # train_test_split:
        # False -> no split, int -> test size, float -> test ratio
        n_test_points = None
        test_size_ratio = None

        if not train_test_split:
            a_test, b_test, c_test = a, b, c
            a_train, b_train, c_train = a, b, c
        else:
            if isinstance(train_test_split, int):
                n_test_points = train_test_split
            elif isinstance(train_test_split, float):
                test_size_ratio = train_test_split
            else:
                raise ValueError(train_test_split)

            # split the data
            a_test, b_test, c_test, a_train, b_train, c_train = split_data(a, b, c, test_size_ratio=test_size_ratio,
                                                                           shuffle=True,
                                                                           n_test_points=n_test_points)
        if a_aux is not None and c_aux is not None:
            if b_aux is not None:
                raise ValueError('Only A/C or B/C data can be auxiliary')
            reg_a = a_aux
            reg_b = b_train
            reg_c_b = c_train
            reg_c_a = c_aux
            reg_b_for_a = None
        elif b_aux is not None and c_aux is not None:
            if a_aux is not None:
                raise ValueError('Only A/C or B/C data can be auxiliary')
            reg_a = a_train
            reg_b = b_aux
            reg_c_b = c_aux
            reg_c_a = c_train
            reg_b_for_a = b_train
        else:
            reg_a, reg_b, reg_c_b, reg_c_a = a_train, b_train, c_train, None
            reg_b_for_a = b_train
        return a_test, b_test, c_test, reg_a, reg_b, reg_c_b, reg_c_a, reg_b_for_a


class KCIMeasure(DependenceMeasure):
    def __init__(self, kernel_a, kernel_ca, kernel_b, kernel_cb,
                 kernel_ca_args, kernel_a_args, kernel_cb_args, kernel_b_args, biased=True,
                 half_split_ca_estimator=False, half_split_cb_estimator=False,
                 circe_a=False, circe_b=False):
        super().__init__()

        self.half_split_ca_estimator = half_split_ca_estimator
        self.half_split_cb_estimator = half_split_cb_estimator
        self.circe_a = circe_a
        self.circe_b = circe_b
        if circe_a and circe_b:
            raise ValueError('circe_a and circe_b cannot be True simultaneously')
        self.kernel_ca = kernel_ca
        self.kernel_a = kernel_a
        self.kernel_ca_args = kernel_ca_args
        self.kernel_a_args = kernel_a_args
        self.kernel_cb = kernel_cb
        self.kernel_b = kernel_b
        self.kernel_cb_args = kernel_cb_args
        self.kernel_b_args = kernel_b_args
        self.biased = biased
        self._reset_krr()
        self._optimal_split_size = None
        self._optimal_split_size_data_size = None  # to check the previous one was found for the same dataset size

    def _reset_krr(self):
        if self.circe_a:
            self.krr_ca = cme.NoRegression(self.kernel_ca, self.kernel_a, self.kernel_ca_args, self.kernel_a_args)
        elif self.half_split_ca_estimator:
            self.krr_ca = cme.SplitKernelRidgeRegression(self.kernel_ca, self.kernel_a, self.kernel_ca_args,
                                                         self.kernel_a_args)
        else:
            self.krr_ca = cme.KernelRidgeRegression(self.kernel_ca, self.kernel_a, self.kernel_ca_args,
                                                    self.kernel_a_args)

        if self.circe_b:
            self.krr_cb = cme.NoRegression(self.kernel_cb, self.kernel_b, self.kernel_cb_args, self.kernel_b_args)
        elif self.half_split_cb_estimator:
            self.krr_cb = cme.SplitKernelRidgeRegression(self.kernel_cb, self.kernel_b, self.kernel_cb_args,
                                                         self.kernel_b_args)
        else:
            self.krr_cb = cme.KernelRidgeRegression(self.kernel_cb, self.kernel_b, self.kernel_cb_args,
                                                    self.kernel_b_args)
        self._regression_done = False

    def compute_statistic(self, a, b, c, return_matrices=False):
        # testing a _|| b | c
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        K_aa_centered = self.krr_ca.predict_kernel_matrix(c, a)
        K_bb_centered = self.krr_cb.predict_kernel_matrix(c, b)

        # todo: need to choose the one with all dims
        if isinstance(self.krr_cb, cme.KernelRidgeRegression):
            K_cc = self.krr_cb.eval_K_xx(c)
        elif isinstance(self.krr_cb, cme.SplitKernelRidgeRegression):
            K_cc = self.krr_cb.krr_one.eval_K_xx(c)
        else:
            raise NotImplementedError('Currently the code assume krr_cb can compute K_cc')

        K_bb_centered = K_bb_centered * K_cc

        statistic_value = compute_hsic(K_aa_centered, K_bb_centered, self.biased)

        if return_matrices:
            return statistic_value, K_aa_centered, K_bb_centered
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

    def _find_train_test_split(self, a, b, c, n_resamples=100, alpha_ind=0.05, alpha_both=0.05, **krr_kwargs):
        dataset_size = a.shape[0]
        test_size_grid = torch.hstack((torch.arange(max(100, int(0.1 * dataset_size)), int(0.5 * dataset_size), 50),
                                       torch.tensor([int(0.5 * dataset_size)])))
        current_rejection_rate = 1

        if self._regression_done:
            self._reset_krr()

        for n_test_points in test_size_grid.flip(dims=(0,)):
            if current_rejection_rate <= alpha_both:
                return n_test_points

            rejected_splits = torch.zeros(n_resamples)
            for i in range(n_resamples):
                a_test, b_test, c_test, a_train, b_train, c_train = split_data(a, b, c, shuffle=True,
                                                                               n_test_points=n_test_points)
                self.find_regressors(a_train, b_train, c_train,
                                     param_dict_cb=krr_kwargs['param_dict_cb'],
                                     lambda_values_cb=krr_kwargs['lambda_values_cb'],
                                     param_dict_ca=krr_kwargs['param_dict_ca'],
                                     lambda_values_ca=krr_kwargs['lambda_values_ca'],
                                     verbose=krr_kwargs['verbose'], cpu_solver=krr_kwargs['cpu_solver'],
                                     cpu_dtype=krr_kwargs['cpu_dtype'])
                Kaa = self.krr_ca.predict_kernel_matrix(c, a)
                Kbb = self.krr_cb.predict_kernel_matrix(c, b)
                Kcc = self.krr_ca.eval_K_xx(c)
                kci_ac = compute_hsic(Kaa, Kcc, biased=self.biased)
                pval_ac = self.compute_pval(kci_ac, K=Kaa, L=Kcc, pval_approx_type=krr_kwargs['pval_approx_type'],
                                            n_samples=krr_kwargs['n_samples'], chunk_size=krr_kwargs['chunk_size'])
                kci_bc = compute_hsic(Kbb, Kcc, biased=self.biased)
                pval_bc = self.compute_pval(kci_bc, K=Kbb, L=Kcc, pval_approx_type=krr_kwargs['pval_approx_type'],
                                            n_samples=krr_kwargs['n_samples'], chunk_size=krr_kwargs['chunk_size'])

                # P(A and B) <= P(A) or P(B), which we cap at alpha
                rejected_splits[i] = (pval_ac <= alpha_ind) and (pval_bc <= alpha_ind)

                # making sure this procedure doesn't change the regression state
                self._reset_krr()

            current_rejection_rate = rejected_splits.mean()

        return test_size_grid[0]

    def _split_data(self, a, b, c, a_aux=None, b_aux=None, c_aux=None, param_dict_cb=None, lambda_values_cb=None,
                    param_dict_ca=None, lambda_values_ca=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128,
                    pval_approx_type='wild', train_test_split='auto', n_samples=1000, chunk_size=500,
                    alpha_ind=0.05, alpha_both=0.05):
        # train_test_split:
        # False -> no split, 'auto' -> auto, int -> test size, float -> test ratio
        n_test_points = None
        test_size_ratio = None

        if not train_test_split:
            a_test, b_test, c_test = a, b, c
            a_train, b_train, c_train = a, b, c
        else:
            if train_test_split == 'auto':
                if self._optimal_split_size is None:
                    n_test_points = self._find_train_test_split(a, b, c, n_resamples=100,
                                                                alpha_ind=alpha_ind, alpha_both=alpha_both,
                                                                param_dict_cb=param_dict_cb,
                                                                lambda_values_cb=lambda_values_cb,
                                                                param_dict_ca=param_dict_ca,
                                                                lambda_values_ca=lambda_values_ca, verbose=verbose,
                                                                cpu_solver=cpu_solver, cpu_dtype=cpu_dtype,
                                                                pval_approx_type=pval_approx_type,
                                                                n_samples=n_samples, chunk_size=chunk_size)
                    self._optimal_split_size = n_test_points
                    self._optimal_split_size_data_size = a.shape[0]
                else:
                    if self._optimal_split_size_data_size != a.shape[0]:
                        raise ValueError(f'self._optimal_split_size_data_size={self._optimal_split_size_data_size} '
                                         f'!= a.shape[0]={a.shape[0]}; '
                                         f're-instantiate the measure to run on a new dataset.')
                    n_test_points = self._optimal_split_size
            elif isinstance(train_test_split, int):
                n_test_points = train_test_split
            elif isinstance(train_test_split, float):
                test_size_ratio = train_test_split
            else:
                raise ValueError(train_test_split)

            # split the data
            a_test, b_test, c_test, a_train, b_train, c_train = split_data(a, b, c, test_size_ratio=test_size_ratio,
                                                                           shuffle=True,
                                                                           n_test_points=n_test_points)
        if a_aux is not None and c_aux is not None:
            if b_aux is not None:
                raise ValueError('Only A/C or B/C data can be auxiliary')
            reg_a = a_aux
            reg_b = b_train
            reg_c_b = c_train
            reg_c_a = c_aux
        elif b_aux is not None and c_aux is not None:
            if a_aux is not None:
                raise ValueError('Only A/C or B/C data can be auxiliary')
            reg_a = a_train
            reg_b = b_aux
            reg_c_b = c_aux
            reg_c_a = c_train
        else:
            reg_a, reg_b, reg_c_b, reg_c_a = a_train, b_train, c_train, None
        return a_test, b_test, c_test, reg_a, reg_b, reg_c_b, reg_c_a

    def test(self, a, b, c, a_aux=None, b_aux=None, c_aux=None, param_dict_cb=None, lambda_values_cb=None,
             param_dict_ca=None, lambda_values_ca=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128,
             pval_approx_type='wild', train_test_split='auto', n_samples=1000, chunk_size=500,
             alpha_ind=0.05, alpha_both=0.05, **ignored):
        # train_test_split:
        # None -> no split, 'auto' -> auto, int -> test size, float -> test ratio
        self._reset_krr()
        a_test, b_test, c_test, reg_a, reg_b, reg_c_b, reg_c_a = self._split_data(
            a, b, c, a_aux, b_aux, c_aux, param_dict_cb, lambda_values_cb,
            param_dict_ca, lambda_values_ca, verbose, cpu_solver, cpu_dtype,
            pval_approx_type, train_test_split, n_samples, chunk_size, alpha_ind, alpha_both)

        self.find_regressors(reg_a, reg_b, c_b=reg_c_b, c_a=reg_c_a,
                             param_dict_cb=param_dict_cb, lambda_values_cb=lambda_values_cb,
                             param_dict_ca=param_dict_ca, lambda_values_ca=lambda_values_ca,
                             verbose=verbose, cpu_solver=cpu_solver,
                             cpu_dtype=cpu_dtype)
        # test
        stat, K, L = self.compute_statistic(a_test, b_test, c_test, return_matrices=True)
        pval = self.compute_pval(stat, K=K, L=L, pval_approx_type=pval_approx_type,
                                 n_samples=n_samples, chunk_size=chunk_size)
        return pval


class GCMMeasure(DependenceMeasure):
    def __init__(self, kernel_ca, kernel_cb, kernel_ca_args, kernel_cb_args):
        super().__init__()
        self.kernel_cb_args = kernel_cb_args
        self.kernel_b_args = {'n': 1}
        self.kernel_ca_args = kernel_ca_args
        self.kernel_a_args = {'n': 1}

        self.kernel_a = 'linear'
        self.kernel_ca = kernel_ca
        self.kernel_b = 'linear'
        self.kernel_cb = kernel_cb

        self._reset_krr()

    def _reset_krr(self):
        self.krr_ca = cme.KernelRidgeRegression(self.kernel_ca, self.kernel_a, self.kernel_ca_args,
                                                self.kernel_a_args)
        self.krr_cb = cme.KernelRidgeRegression(self.kernel_cb, self.kernel_b, self.kernel_cb_args,
                                                self.kernel_b_args)
        self._regression_done = False

    def compute_statistic(self, a, b, c, **ignored):
        # testing a _|| b | c
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')
        residual_a = a - self.krr_ca.predict(c)
        residual_b = b - self.krr_cb.predict(c)

        r_vals = residual_a[:, :, None] * residual_b[:, None, :]

        n = r_vals.shape[0]
        gcm_mean = r_vals.mean(dim=0)
        tau_d = ((r_vals ** 2).mean(dim=0) - gcm_mean ** 2) ** 0.5

        statistic_value = np.sqrt(n) * gcm_mean / tau_d
        statistic_value = torch.abs(statistic_value).max()

        if a.shape[1] * b.shape[1] == 1:
            return statistic_value, None

        r_vals = r_vals.view((n, -1)).T
        gcm_mean = gcm_mean.view((-1, 1))
        sigma_half = (r_vals - gcm_mean) / np.sqrt(n) / tau_d.view((-1, 1))

        return statistic_value, sigma_half

    def compute_pval(self, statistic_value, sigma_half, **kwargs):
        return pval_computations.compute_gcm_pval(statistic_value, sigma_half,
                                                  n_samples=kwargs['n_samples'] if 'n_samples' in kwargs else 1000)

    def test(self, a, b, c, a_aux=None, b_aux=None, c_aux=None, param_dict_cb=None, lambda_values_cb=None,
             param_dict_ca=None, lambda_values_ca=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128,
             train_test_split=False, n_samples=1000, **ignored):
        # train_test_split:
        # False -> no split, 'auto' -> auto, int -> test size, float -> test ratio
        self._reset_krr()
        a_test, b_test, c_test, reg_a, reg_b, reg_c_b, reg_c_a, reg_c_for_a = self._split_data(
            a, b, c, a_aux, b_aux, c_aux, train_test_split)

        self.find_regressors(reg_a, reg_b, c_b=reg_c_b, c_a=reg_c_a,
                             param_dict_cb=param_dict_cb, lambda_values_cb=lambda_values_cb,
                             param_dict_ca=param_dict_ca, lambda_values_ca=lambda_values_ca,
                             verbose=verbose, cpu_solver=cpu_solver,
                             cpu_dtype=cpu_dtype)
        # test
        stat_val, sigma_half = self.compute_statistic(a_test, b_test, c_test)
        pval = self.compute_pval(stat_val, sigma_half, n_samples=n_samples)
        return pval


class RBPT2Measure(DependenceMeasure):
    def __init__(self, kernel_w, kernel_c, kernel_w_args, kernel_c_args, apply_bias_correction=False):
        super().__init__()

        self.kernel_w = kernel_w
        self.kernel_c = kernel_c
        self.kernel_w_args = kernel_w_args
        self.kernel_c_args = kernel_c_args
        self.apply_bias_correction = apply_bias_correction

        self._reset_krr()

    def _reset_krr(self):
        if self.kernel_w == 'linreg':
            self.g = cme.LinearRegression()
        else:
            self.g = cme.KernelRidgeRegression(self.kernel_w, 'linear', self.kernel_w_args, dict())
        self.h = cme.KernelRidgeRegression(self.kernel_c, 'linear', self.kernel_c_args, dict())

        self._regression_done = False

    def find_regressors(self, a, b, c_b, c_a=None, param_dict_rbpt_w=None, lambda_values_rbpt_w=None,
                        param_dict_rbpt_c=None, lambda_values_rbpt_c=None, verbose=False, cpu_solver=False,
                        cpu_dtype=np.float128, b_a=None, **ignored):
        # testing a _||_ b | c, and b/c is unlabeled
        if c_a is None:
            c_a = c_b
            b_a = b
        if c_a is not None and b_a is None:
            raise ValueError('Unlabelled data should contain both c_a and b_a, but only c_a was given.')

        loo_g = self.g.fit(torch.hstack((b_a, c_a)), a, verbose=verbose, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype,
                           param_dict_x=param_dict_rbpt_w, lambda_values_x=lambda_values_rbpt_w)
        g_predictions = self.g.predict(torch.hstack((b, c_b)))
        loo_h = self.h.fit(c_b, g_predictions, verbose=verbose, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype,
                           param_dict_x=param_dict_rbpt_c, lambda_values_x=lambda_values_rbpt_c)

        self._regression_done = True
        return loo_g, loo_h

    def compute_statistic(self, a, b, c, **ignored):
        # testing a _||_ b | c, and b/c is unlabeled
        if not self._regression_done:
            raise ValueError('Have to run self.find_regressors before computing the statistic!')

        n = a.shape[0]

        g_predictions = self.g.predict(torch.hstack((b, c)))
        h_predictions = self.h.predict(c)

        loss1 = ((a - g_predictions) ** 2).mean(dim=1)
        loss2 = ((a - h_predictions) ** 2).mean(dim=1)

        # not in the original paper, but improves p-values
        if self.apply_bias_correction:
            bias_correction = ((h_predictions - g_predictions) ** 2).mean(dim=1)
        else:
            bias_correction = 0

        statistic_value = loss2 - loss1 + bias_correction
        statistic_value = np.sqrt(n) * statistic_value.mean() / statistic_value.std()
        return statistic_value

    def compute_pval(self, statistic_value, **kwargs):
        return norm_distr.sf(statistic_value.item())

    def test(self, a, b, c, a_aux=None, b_aux=None, c_aux=None, param_dict_rbpt_w=None, lambda_values_rbpt_w=None,
             param_dict_rbpt_c=None, lambda_values_rbpt_c=None, verbose=False, cpu_solver=False, cpu_dtype=np.float128,
             train_test_split=False, **ignored):
        self._reset_krr()
        # train_test_split:
        # False -> no split, 'auto' -> auto, int -> test size, float -> test ratio
        a_test, b_test, c_test, reg_x, reg_z, reg_y_z, reg_y_x, reg_z_for_x = self._split_data(
            a, b, c, a_aux, b_aux, c_aux, train_test_split)

        self.find_regressors(reg_x, reg_z, c_b=reg_y_z, c_a=reg_y_x, b_a=reg_z_for_x,
                             param_dict_rbpt_w=param_dict_rbpt_w, lambda_values_rbpt_w=lambda_values_rbpt_w,
                             param_dict_rbpt_c=param_dict_rbpt_c, lambda_values_rbpt_c=lambda_values_rbpt_c,
                             verbose=verbose, cpu_solver=cpu_solver,
                             cpu_dtype=cpu_dtype)

        # test
        stat_val = self.compute_statistic(a_test, b_test, c_test)
        pval = self.compute_pval(stat_val)
        return pval
