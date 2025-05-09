import torch
import numpy as np
import splitkci.cond_mean_estimation as cme
import os

from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import OneOf
from main_experiment import get_measure
from tasks import load_rat_data


Section('problem_setup', 'model details').params(
    ground_truth=Param(OneOf(['H0', 'H1']), default='H0'),
    dim=Param(int, default=10),
    budget=Param(int, default=400),
    measure=Param(OneOf(['kci', 'circe', 'kci_asplit', 'kci_absplit', 'circe_bsplit', 'gcm',
                         'rbpt2', 'rbpt2_linreg_w', 'rbpt2_ub']),
                  default='kci'),
    ca_kernels=Param(OneOf(['gaussian', 'all']), default='gaussian')
)

Section('pval', 'pval setup').params(
    pval_estimation=Param(OneOf(['gamma', 'wild']), default='wild'),
    n_data_resamples=Param(int, default=1),
    # n_holdout_resamples=Param(int, default=1),
    n_holdout_resamples_start=Param(int, default=0),
    n_holdout_resamples_end=Param(int, default=1),
    n_points_wild_bootstrap=Param(int, default=1000)
)

Section('files', 'save/load').params(
    filename=Param(str, default='test'),
)


def split_data(a_all, b_all, c_all, test_size_ratio, shuffle):
    n_test_points = int(a_all.shape[0] * test_size_ratio)

    if shuffle:
        idx_all = torch.randperm(a_all.shape[0], device=a_all.device)
    else:
        idx_all = torch.arange(a_all.shape[0], device=a_all.device)

    idx_test, idx_train = idx_all[:n_test_points], idx_all[n_test_points:]

    a_test, b_test, c_test = a_all[idx_test], b_all[idx_test], c_all[idx_test]
    a_train, b_train, c_train = a_all[idx_train], b_all[idx_train], c_all[idx_train]

    return a_test, b_test, c_test, a_train, b_train, c_train


@param('problem_setup.ground_truth')
@param('problem_setup.measure')
@param('pval.pval_estimation')
@param('pval.n_holdout_resamples_start')
@param('pval.n_holdout_resamples_end')
@param('pval.n_points_wild_bootstrap')
def run_task(device, dim, budget, test_size_ratio, kernel_a, kernel_ca, kernel_b, kernel_cb,
             kernel_ca_args, kernel_a_args,
             kernel_cb_args, kernel_b_args, param_dict_cb, param_dict_ca,
             *, ground_truth, measure,
             pval_estimation, n_holdout_resamples_start, n_holdout_resamples_end,
             n_points_wild_bootstrap):
    biased = pval_estimation == 'gamma'
    p_accepted_h0 = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start)
    Kaa_list = list()
    Kbb_list = list()
    Kcc_list = list()
    all_loo_ca = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start,
                             2 if ('asplit' in measure or 'absplit' in measure) else 1)
    # all_cme_test_yx = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start,
    #                          2 if ('asplit' in measure or 'absplit' in measure) else 1)
    all_loo_cb = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start,
                             2 if ('bsplit' in measure) else 1)
    # all_cme_test_yz = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start,
    #                          2 if ('bsplit' in measure) else 1)
    # all_cme_test_joint = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start)

    for idx_ho_sample in range(n_holdout_resamples_start, n_holdout_resamples_end):
        # notes: a lot of things are hard-coded; e.g. rbpt would only work with the current setup but masking might fail
        # also masking only works with 4dim y
        # and only for 3k saved points

        # the points are already shuffled
        a_all, b_all, c_all = load_rat_data(idx_ho_sample, 3000, ground_truth, dim, device)
        a_all, b_all, c_all = a_all[:budget], b_all[:budget], c_all[:budget]

        # test:         a    b      c
        # x-regression: a_ho b_ho_a c_ho_a
        # z-regression: _    b_ho   c_ho_b

        # joint: a-reg and b-reg share the data
        # separate: a-reg and b-reg are independent
        # no train test split: test and a-reg are combined

        a_test, b_test, c_test, a_train, b_train, c_train = split_data(a_all, b_all, c_all, test_size_ratio,
                                                                       shuffle=True)

        kci = get_measure(measure, kernel_a, kernel_b, kernel_ca, kernel_cb, kernel_ca_args, kernel_cb_args,
                          kernel_b_args, kernel_a_args, biased,
                          half_split_ca_estimator='asplit' in measure or 'absplit' in measure,
                          half_split_cb_estimator='bsplit' in measure)

        loo_ca, loo_cb = kci.find_regressors(a_train, b_train, c_b=c_train, param_dict_cb=param_dict_cb, verbose=False,
                                             cpu_solver=True, c_a=c_train, param_dict_ca=param_dict_ca,
                                             z_x=b_train, param_dict_rbpt_w=param_dict_cb if isinstance(kernel_ca, list) else None,
                                             param_dict_rbpt_y=param_dict_ca)

        Kaa_list.append(kci.krr_ca.predict_kernel_matrix(c_test, a_test))
        Kbb_list.append(kci.krr_cb.predict_kernel_matrix(c_test, b_test))
        # Kxx_list_train.append(kci.krr_ca.predict_kernel_matrix(c_train, a_train))
        # Kzz_list_train.append(kci.krr_cb.predict_kernel_matrix(c_train, b_train))
        if isinstance(kci.krr_cb, cme.SplitKernelRidgeRegression):
            Kcc_list.append(kci.krr_cb.krr_one.eval_K_xx(c_test))
            # Kyy_list_train.append(kci.krr_cb.krr_one.eval_K_xx(c_train))
        else:
            Kcc_list.append(kci.krr_cb.eval_K_xx(c_test))
            # Kyy_list_train.append(kci.krr_cb.eval_K_xx(c_train))

        all_loo_ca[idx_ho_sample - n_holdout_resamples_start] = torch.tensor(loo_ca)
        all_loo_cb[idx_ho_sample - n_holdout_resamples_start] = torch.tensor(loo_cb)

        if measure == 'gcm':
            kci_val, sigma_half = kci.compute_statistic(a_test, b_test, c_test)

            p_accepted_h0[idx_ho_sample - n_holdout_resamples_start] = \
                kci.compute_pval(kci_val, sigma_half, n_samples=n_points_wild_bootstrap)
        elif 'rbpt2' in measure:
            kci_val = kci.compute_statistic(a_test, b_test, c_test)

            p_accepted_h0[idx_ho_sample - n_holdout_resamples_start] = kci.compute_pval(kci_val)
        else:
            kci_val, K, L = kci.compute_statistic(a_test, b_test, c_test, return_matrices=True)

            p_accepted_h0[idx_ho_sample - n_holdout_resamples_start] = \
                kci.compute_pval(kci_val, pval_approx_type=pval_estimation, K=K, L=L,
                                 n_samples=n_points_wild_bootstrap, chunk_size=500)

    return {'pval': p_accepted_h0, 'loo_ca': all_loo_ca, 'loo_cb': all_loo_cb,
            'Kaa_list': Kaa_list, 'Kbb_list': Kbb_list, 'Kcc_list': Kcc_list,
            }


@param('problem_setup.dim')
@param('problem_setup.budget')
@param('problem_setup.ca_kernels')
@param('pval.n_holdout_resamples_start')
@param('pval.n_holdout_resamples_end')
def main_kci_partial(saved_file_path, dim, budget, ca_kernels, n_holdout_resamples_start, n_holdout_resamples_end):
    kernel_a = 'gaussian_precision'
    kernel_b = 'gaussian_precision'
    kernel_cb = 'gaussian_precision'
    kernel_a_args = {'gamma': 1.0}
    kernel_b_args = {'gamma': 1.0}
    kernel_cb_args = {'gamma': 1.0}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if ca_kernels == 'gaussian':
        param_dict_ca = {'gamma': torch.tensor([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])}
        kernel_ca = 'gaussian_precision'
    elif ca_kernels == 'all':
        param_dict_ca = {
            'gaussian_precision': {'gamma': torch.tensor([0.1, 0.2, 0.5, 1.0, 1.5, 2.0]),
                                   'int_bin_mask': torch.tensor([float(int('0b1111', 2)),
                                                                 float(int('0b1100', 2)),
                                                                 float(int('0b0011', 2))])},
        }

        kernel_ca = [
            'gaussian_precision',
        ]
    else:
        raise NotImplementedError(f'ca_kernels={ca_kernels} has to be gaussian or all')
    kernel_ca_args = {'gamma': 1.0}

    param_dict_cb = {'gamma': torch.tensor([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])}

    test_size_ratio_grid = np.linspace(0.1, 0.9, 9)

    for test_size_ratio in test_size_ratio_grid:
        final_saved_file_path = saved_file_path + \
                                f'seeds{n_holdout_resamples_start}_{n_holdout_resamples_end}_' + \
                                f'd{dim}_r{test_size_ratio:.2f}.pt'

        if not os.path.isfile(final_saved_file_path):
            task_result = run_task(device, dim, budget, test_size_ratio, kernel_a, kernel_ca, kernel_b,
                                   kernel_cb, kernel_ca_args, kernel_a_args, kernel_cb_args,
                                   kernel_b_args, param_dict_cb, param_dict_ca)

            torch.save(task_result, final_saved_file_path)


@param('files.filename')
def main(filename):
    saved_file_path = os.path.join(os.environ['SCRATCH'], 'splitkci_testing_results/budget_experiment', f'{filename}')
    main_kci_partial(saved_file_path)


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Testing on toy problems')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
