import torch
from splitkci import dependence_measures
import os

from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import OneOf
from tasks import load_rat_data, get_abc_randn


Section('problem_setup', 'model details').params(
    ground_truth=Param(OneOf(['H0', 'H1']), default='H0'),
    dim=Param(int, default=10),
    measure=Param(OneOf(['kci', 'circe', 'kci_asplit', 'kci_absplit', 'circe_bsplit', 'gcm',
                         'rbpt2', 'rbpt2_linreg_w', 'rbpt2_ub', 'rbpt2_ub_linreg_w']),
                  default='kci'),
    abc_holdout_sampling=Param(OneOf(['joint', 'separate']), default='separate'),
    ca_kernels=Param(OneOf(['gaussian', 'all']), default='gaussian'),
    abc_budget=Param(int, default=200),
    min_grid_points=Param(int, default=100),
    train_test_split=Param(str, default='False'),
    control=Param(str, default='False'),
    task=Param(OneOf(['ratinabox', 'randn']), default='ratinabox')
)

Section('pval', 'pval setup').params(
    pval_estimation=Param(OneOf(['gamma', 'wild']), default='wild'),
    n_data_resamples=Param(int, default=1),
    n_holdout_resamples_start=Param(int, default=0),
    n_holdout_resamples_end=Param(int, default=1),
    n_points_wild_bootstrap=Param(int, default=1000),
    split_alpha_ind=Param(float, default=0.05),
    split_alpha_both=Param(float, default=0.05)
)

Section('files', 'save/load').params(
    filename=Param(str, default='test'),
)


def get_measure(measure, kernel_a, kernel_b, kernel_ca, kernel_cb, kernel_ca_args, kernel_cb_args, kernel_b_args,
                kernel_a_args, biased, half_split_ca_estimator, half_split_cb_estimator):
    if 'circe' in measure:
        if (measure != 'circe') and (measure != 'circe_bsplit'):
            raise ValueError(measure)
        ci_test = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb,
                                                 kernel_ca_args, kernel_a_args, kernel_cb_args, kernel_b_args,
                                                 biased, half_split_ca_estimator, half_split_cb_estimator,
                                                 circe_a=True)
    elif 'kci' in measure:
        ci_test = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb,
                                                 kernel_ca_args, kernel_a_args, kernel_cb_args, kernel_b_args,
                                                 biased, half_split_ca_estimator, half_split_cb_estimator)
    elif measure == 'gcm':
        ci_test = dependence_measures.GCMMeasure(kernel_ca, kernel_cb,
                                                 kernel_ca_args, kernel_cb_args)
    elif measure == 'rbpt2' or measure == 'rbpt2_ub':
        if isinstance(kernel_ca, list):
            ci_test = dependence_measures.RBPT2Measure(kernel_w='gaussian_precision', kernel_c=kernel_ca,
                                                       kernel_w_args={'gamma': 1.0},
                                                       kernel_c_args=kernel_ca_args,
                                                       apply_bias_correction='rbpt2_ub' in measure)
        else:
            ci_test = dependence_measures.RBPT2Measure(kernel_w='linear', kernel_c=kernel_ca,
                                                       kernel_w_args={'n': 1}, kernel_c_args=kernel_ca_args,
                                                       apply_bias_correction='rbpt2_ub' in measure)
    elif measure == 'rbpt2_linreg_w' or measure == 'rbpt2_ub_linreg_w':
        ci_test = dependence_measures.RBPT2Measure(kernel_w='linreg', kernel_c=kernel_ca,
                                                   kernel_w_args={'gamma': 1.0},
                                                   kernel_c_args=kernel_ca_args,
                                                   apply_bias_correction='rbpt2_ub' in measure)
    else:
        raise NotImplementedError(f'measure={measure} has to be kci/circe or variations')
    return ci_test


@param('problem_setup.task')
def sample_task_data(seed, ground_truth, dim, device, task):
    if task == 'ratinabox':
        return load_rat_data(seed, 3000, ground_truth, dim, device)
    elif task == 'randn':
        return get_abc_randn(3000, ground_truth, dim, device)


@param('problem_setup.ground_truth')
@param('problem_setup.measure')
@param('problem_setup.abc_holdout_sampling')
@param('problem_setup.train_test_split')
@param('problem_setup.control')
@param('pval.pval_estimation')
@param('pval.n_data_resamples')
@param('pval.n_holdout_resamples_start')
@param('pval.n_holdout_resamples_end')
@param('pval.n_points_wild_bootstrap')
@param('pval.split_alpha_ind')
@param('pval.split_alpha_both')
def run_task(device, dim, n_abc_points, n_aux_bc_points,
             kernel_a, kernel_ca, kernel_b, kernel_cb, kernel_ca_args, kernel_a_args,
             kernel_cb_args, kernel_b_args, param_dict_cb, param_dict_ca,
             *, ground_truth, measure, abc_holdout_sampling,
             train_test_split, control,
             pval_estimation, n_data_resamples, n_holdout_resamples_start, n_holdout_resamples_end,
             n_points_wild_bootstrap, split_alpha_ind, split_alpha_both):
    biased = pval_estimation == 'gamma'
    p_accepted_h0 = torch.zeros(n_holdout_resamples_end - n_holdout_resamples_start, n_data_resamples)
    assert n_data_resamples == 1, "No resamples for the generated data"
    idx_sample = 0

    if control == 'True':
        print('Running a control experiment')
        # check it's joint
        # equalize points
        assert abc_holdout_sampling == 'joint'
        train_test_split = 0.5
        n_abc_points = 2 * n_abc_points
    else:
        if train_test_split != 'False' and train_test_split != 'auto':
            if float(train_test_split) > 1:
                train_test_split = int(train_test_split)
            else:
                train_test_split = float(train_test_split)
        elif train_test_split == 'False' or (train_test_split == 'auto' and measure != 'kci_absplit'):
            train_test_split = False
    print(f'Train/test split: {train_test_split}, n_abc: {n_abc_points}, n_aux: {n_aux_bc_points}')

    for idx_ho_sample in range(n_holdout_resamples_start, n_holdout_resamples_end):
        # notes: a lot of things are hard-coded; e.g. rbpt would only work with the current setup but masking might fail
        # also masking only works with 4dim y
        # and only for 3k saved points

        # the points are already shuffled
        a_all, b_all, c_all = sample_task_data(idx_ho_sample, ground_truth, dim, device)
        n_total_points = n_abc_points + n_aux_bc_points
        a_all, b_all, c_all = a_all[:n_total_points], b_all[:n_total_points], c_all[:n_total_points]

        # test:         x    z      y
        # x-regression: x_ho z_ho_x y_ho_x
        # z-regression: _    z_ho   y_ho_z

        # joint: x-reg and z-reg share the data
        # separate: x-reg and z-reg are independent
        # no train test split: test and x-reg are combined

        ci_test = get_measure(measure, kernel_a, kernel_b, kernel_ca, kernel_cb, kernel_ca_args, kernel_cb_args,
                              kernel_b_args, kernel_a_args, biased,
                              half_split_ca_estimator='asplit' in measure or 'absplit' in measure,
                              half_split_cb_estimator='bsplit' in measure)

        if abc_holdout_sampling == 'separate':
            b_aux = b_all[n_abc_points:].clone()
            c_aux = c_all[n_abc_points:].clone()
        else:
            b_aux = None
            c_aux = None
        a_all, b_all, c_all = a_all[:n_abc_points], b_all[:n_abc_points], c_all[:n_abc_points]

        pval = ci_test.test(a_all, b_all, c_all, a_aux=None, b_aux=b_aux, c_aux=c_aux,
                            param_dict_cb=param_dict_cb,
                            param_dict_ca=param_dict_ca,
                            verbose=False, cpu_solver=True,
                            pval_approx_type=pval_estimation, train_test_split=train_test_split,
                            n_samples=n_points_wild_bootstrap, chunk_size=500,
                            alpha_ind=split_alpha_ind, alpha_both=split_alpha_both,
                            param_dict_rbpt_w=param_dict_cb if isinstance(kernel_ca, list) else None,
                            param_dict_rbpt_c=param_dict_ca)

        p_accepted_h0[idx_ho_sample - n_holdout_resamples_start, idx_sample] = pval

    return p_accepted_h0


@param('problem_setup.dim')
@param('problem_setup.abc_holdout_sampling')
@param('problem_setup.abc_budget')
@param('problem_setup.min_grid_points')
@param('problem_setup.ca_kernels')
@param('problem_setup.train_test_split')
@param('problem_setup.control')
@param('pval.n_holdout_resamples_start')
@param('pval.n_holdout_resamples_end')
def run_test(saved_file_path, dim, abc_holdout_sampling, abc_budget, min_grid_points,
             ca_kernels,
             train_test_split, control, n_holdout_resamples_start, n_holdout_resamples_end):
    # setting kernel parameters
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

    # setting point grids
    points_grid = torch.linspace(min_grid_points, 1000, (1000 - min_grid_points) // 100 + 1).int()

    for n_points in points_grid:
        # fixed_budget: ABC for separate
        if abc_holdout_sampling == 'joint':
            n_abc_points = n_points
            n_aux_bc_points = 0
        else:
            n_abc_points = abc_budget
            n_aux_bc_points = n_points

        if control == 'True':
            final_saved_file_path = saved_file_path + f'seeds{n_holdout_resamples_start}_{n_holdout_resamples_end}_' + \
                                    f'd{dim}_abc{n_abc_points}_aux{n_aux_bc_points}_control.pt'
        else:
            final_saved_file_path = saved_file_path + f'seeds{n_holdout_resamples_start}_{n_holdout_resamples_end}_' + \
                                    f'd{dim}_abc{n_abc_points}_aux{n_aux_bc_points}_ttsplit_{train_test_split}.pt'

        if not os.path.isfile(final_saved_file_path):
            task_result = run_task(device, dim, n_abc_points, n_aux_bc_points,
                                   kernel_a, kernel_ca, kernel_b,
                                   kernel_cb, kernel_ca_args, kernel_a_args, kernel_cb_args,
                                   kernel_b_args, param_dict_cb, param_dict_ca)

            torch.save(task_result, final_saved_file_path)


@param('files.filename')
@param('problem_setup.task')
@param('pval.pval_estimation')
def main(filename, task, pval_estimation):
    if pval_estimation == 'gamma':
        filename += '_gamma'
    saved_file_path = os.path.join(os.environ['SCRATCH'], f'splitkci_testing_results/{task}', f'{filename}')
    run_test(saved_file_path)


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
