import torch
import numpy as np
import dependence_measures
import os
import toy_tasks

from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf


Section('problem_setup', 'model details').params(
    task=Param(OneOf(toy_tasks.__all__), default='randn'),
    ground_truth=Param(OneOf(['H0', 'H1']), default='H0'),
    min_dim=Param(int, default=2),
    max_dim=Param(int, default=2),
    measure=Param(OneOf(['kci', 'circe', 'kci_xsplit', 'kci_xzsplit', 'circe_zsplit', 'gcm',
                         'rbpt2', 'rbpt2_linreg_w', 'rbpt2_ub']),
                  default='kci'),
    xzy_holdout_sampling=Param(OneOf(['joint', 'separate']), default='separate'),
    yx_kernels=Param(OneOf(['gaussian', 'all']), default='gaussian'),
    n_points=Param(int, default=100),
    n_xy_points=Param(int, default=100),
    min_n_zy_points=Param(int, default=100),
    max_n_zy_points=Param(int, default=1000),
    rbpt_c=Param(float, default=0.1),
    rbpt_gamma=Param(float, default=0.01),
    rbpt_seed=Param(int, default=1)
)

Section('pval', 'pval setup').params(
    pval_estimation=Param(OneOf(['gamma', 'wild']), default='wild'),
    n_data_resamples=Param(int, default=1),
    n_holdout_resamples=Param(int, default=1),
    n_points_wild_bootstrap=Param(int, default=1000)
)

Section('files', 'save/load').params(
    filename=Param(str, default='test'),
)


@param('problem_setup.task')
@param('problem_setup.ground_truth')
@param('problem_setup.measure')
@param('problem_setup.xzy_holdout_sampling')
@param('problem_setup.n_points')
@param('problem_setup.n_xy_points')
@param('problem_setup.rbpt_c')
@param('problem_setup.rbpt_gamma')
@param('problem_setup.rbpt_seed')
@param('pval.pval_estimation')
@param('pval.n_data_resamples')
@param('pval.n_holdout_resamples')
@param('pval.n_points_wild_bootstrap')
def run_task(device, dim, n_zy_points, kernel_x, kernel_yx, kernel_z, kernel_yz, kernel_yx_args, kernel_x_args,
             kernel_yz_args, kernel_z_args, param_dict_yz, param_dict_yx,
             *, task, ground_truth, measure, xzy_holdout_sampling, n_points, n_xy_points, rbpt_c, rbpt_gamma, rbpt_seed,
             pval_estimation, n_data_resamples, n_holdout_resamples, n_points_wild_bootstrap):
    get_xzy = getattr(toy_tasks, task)
    biased = pval_estimation == 'gamma'
    p_accepted_h0 = torch.zeros(n_holdout_resamples, n_data_resamples)

    for idx_ho_sample in range(n_holdout_resamples):
        if xzy_holdout_sampling == 'joint':
            x_ho, z_ho, y_ho_z = get_xzy(n_zy_points, ground_truth, dim, device=device,
                                         c=rbpt_c, gamma=rbpt_gamma, seed=rbpt_seed)
            y_ho_x = y_ho_z.clone()
            z_ho_x = z_ho.clone()
        elif xzy_holdout_sampling == 'separate':
            _, z_ho, y_ho_z = get_xzy(n_zy_points, ground_truth, dim, device=device,
                                      c=rbpt_c, gamma=rbpt_gamma, seed=rbpt_seed)

            x_ho, z_ho_x, y_ho_x = get_xzy(n_xy_points, ground_truth, dim, device=device,
                                           c=rbpt_c, gamma=rbpt_gamma, seed=rbpt_seed)
        else:
            raise NotImplementedError(f'xzy_holdout_sampling={xzy_holdout_sampling} has to be joint or separate')

        if get_xzy == toy_tasks.get_xzy_randn_nl:
            kernel_x_args['sigma2'] = (x_ho.norm(dim=1) ** 2).mean()
            kernel_z_args['sigma2'] = (z_ho.norm(dim=1) ** 2).mean()
        elif get_xzy == toy_tasks.get_xzy_rbpt:
            kernel_x_args['sigma2'] = x_ho.std() ** 2
            kernel_z_args['sigma2'] = z_ho.std() ** 2

        if 'circe' in measure:
            kci = dependence_measures.CirceMeasure(kernel_x, kernel_z, kernel_yz,
                                                   kernel_yz_args, kernel_z_args, kernel_x_args,
                                                   biased=biased)
        elif 'kci' in measure:
            kci = dependence_measures.KCIMeasure(kernel_x, kernel_yx, kernel_z, kernel_yz,
                                                 kernel_yx_args, kernel_x_args, kernel_yz_args, kernel_z_args,
                                                 biased=biased)
        elif measure == 'gcm':
            kci = dependence_measures.GCMMeasure(kernel_yx, kernel_yz,
                                                 kernel_yx_args, kernel_yz_args)
        elif measure == 'rbpt2' or measure == 'rbpt2_ub':
            if isinstance(kernel_yx, list):
                kci = dependence_measures.RBPT2Measure(kernel_w='gaussian', kernel_y=kernel_yx,
                                                       kernel_w_args={'sigma2': 1.0},
                                                       kernel_y_args=kernel_yx_args)
            else:
                kci = dependence_measures.RBPT2Measure(kernel_w='linear', kernel_y=kernel_yx,
                                                       kernel_w_args={'n': 1}, kernel_y_args=kernel_yx_args)
        elif measure == 'rbpt2_linreg_w':
            kci = dependence_measures.RBPT2Measure(kernel_w='linreg', kernel_y=kernel_yx,
                                                   kernel_w_args={'sigma2': 1.0},
                                                   kernel_y_args=kernel_yx_args)
        else:
            raise NotImplementedError(f'measure={measure} has to be kci/circe or variations')

        kci.find_regressors(x_ho, z_ho, y_z=y_ho_z, param_dict_yz=param_dict_yz, verbose=False,
                            cpu_solver=True, y_x=y_ho_x, param_dict_yx=param_dict_yx,
                            half_split_yx_estimator='xsplit' in measure or 'xzsplit' in measure,
                            half_split_yz_estimator='zsplit' in measure,
                            z_x=z_ho_x, param_dict_rbpt_w=param_dict_yz if isinstance(kernel_yx, list) else None,
                            param_dict_rbpt_y=param_dict_yx)
        for idx_sample in range(n_data_resamples):
            x, z, y = get_xzy(n_points, ground_truth, dim, device=device,
                              rbpt_c=rbpt_c, rbpt_gamma=rbpt_gamma, seed=rbpt_seed)

            if measure == 'gcm':
                kci_val, sigma_half = kci.compute_statistic(x, z, y)

                p_accepted_h0[idx_ho_sample, idx_sample] = \
                    kci.compute_pval(kci_val, sigma_half, n_samples=n_points_wild_bootstrap)
            elif 'rbpt2' in measure:
                kci_val = kci.compute_statistic(x, z, y)

                p_accepted_h0[idx_ho_sample, idx_sample] = kci.compute_pval(kci_val)
            else:
                kci_val, K, L = kci.compute_statistic(x, z, y, return_matrices=True)

                p_accepted_h0[idx_ho_sample, idx_sample] = \
                    kci.compute_pval(kci_val, pval_approx_type=pval_estimation, K=K, L=L,
                                     n_samples=n_points_wild_bootstrap)

    return p_accepted_h0


@param('problem_setup.min_dim')
@param('problem_setup.max_dim')
@param('problem_setup.min_n_zy_points')
@param('problem_setup.max_n_zy_points')
@param('problem_setup.yx_kernels')
def main_kci_partial(saved_file_path, min_dim, max_dim, min_n_zy_points, max_n_zy_points, yx_kernels):
    kernel_x = 'gaussian'
    kernel_z = 'gaussian'
    kernel_yz = 'gaussian'
    kernel_x_args = {'sigma2': 1.0}
    kernel_yz_args = {'sigma2': 1.0}
    kernel_z_args = {'sigma2': 1.0}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if yx_kernels == 'gaussian':
        param_dict_yx = {'sigma2': torch.tensor([0.5, 1.0, 2.0, 5.0])}
        kernel_yx = 'gaussian'
    elif yx_kernels == 'all':
        param_dict_yx = {
            'gaussian': {'sigma2': torch.tensor([0.5, 1.0, 2.0, 5.0])},
            'poly_decaying': {'alpha': torch.linspace(0.0, 1.0, 6),
                              'n': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            'poly': {'c': torch.linspace(0.1, 1.0, 10),
                     'n': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        }

        kernel_yx = [
            'gaussian',
            'poly_decaying',
            'poly'
        ]
    else:
        raise NotImplementedError(f'yx_kernels={yx_kernels} has to be gaussian or all')
    kernel_yx_args = {'sigma2': 1.0, 'n': 3}

    param_dict_yz = {'sigma2': torch.tensor([0.5, 1.0, 2.0, 5.0])}

    if max_n_zy_points > 1000:
        if min_n_zy_points < 1000:
            zy_points_grid = torch.linspace(min_n_zy_points, 1000,
                                            (1000 - min_n_zy_points) // 100 + 1).int()
            min_n_zy_points = 2000
            zy_points_grid = torch.cat((zy_points_grid, torch.linspace(min_n_zy_points, max_n_zy_points,
                                            (max_n_zy_points - min_n_zy_points) // 1000 + 1).int()))
        else:
            zy_points_grid = torch.linspace(min_n_zy_points, max_n_zy_points,
                                            (max_n_zy_points - min_n_zy_points) // 1000 + 1).int()
    else:
        zy_points_grid = torch.linspace(min_n_zy_points, max_n_zy_points,
                                        (max_n_zy_points - min_n_zy_points) // 100 + 1).int()

    for idx_dim, dim in enumerate(np.arange(min_dim, max_dim + 1)):
        for idx_zy, n_zy_points in enumerate(zy_points_grid):
            final_saved_file_path = saved_file_path + f'd{dim}_nzy{n_zy_points}.pt'
            if not os.path.isfile(final_saved_file_path):
                task_result = run_task(device, dim, n_zy_points, kernel_x, kernel_yx, kernel_z,
                                                          kernel_yz, kernel_yx_args, kernel_x_args, kernel_yz_args,
                                                          kernel_z_args, param_dict_yz, param_dict_yx)

                torch.save(task_result, final_saved_file_path)


@param('files.filename')
def main(filename):
    saved_file_path = os.path.join(os.environ['HOME'], 'testing_results', f'{filename}')
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
