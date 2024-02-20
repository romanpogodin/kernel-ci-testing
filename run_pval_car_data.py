# adapted from the RBPT paper https://github.com/felipemaiapolo/cit/tree/main
import os
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import torch
import dependence_measures
from tqdm import tqdm
import gc


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


Section('test', 'model details').params(
    measure=Param(OneOf(['kci', 'kci_xsplit', 'circe']), default='kci')
)


@param('test.measure')
def run_test(measure):
    states = ['ca', 'il', 'mo', 'tx']
    pvals = []

    random_state = 42
    # np.random.seed(random_state)

    for s in tqdm(states):
        # assumes a cloned RBPT repo in $HOME
        data = pd.read_csv(os.path.join(os.environ['HOME'],
                                        'cit/data/car-insurance-public/data/' + s + '-per-zip.csv'))

        Z = np.array(data.state_risk).reshape((-1, 1))
        Y = np.array(data.combined_premium).reshape((-1, 1))
        X = (1 * np.array(data.minority)).reshape((-1, 1))

        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=.3,
                                                                             random_state=random_state)
        X_train, _, _, Y_train, Z_train_x, Z_train_y = train_test_split(X_train, Y_train, Z_train,
                                                                        test_size=0.5, random_state=random_state * 2)

        ###Fitting models
        # regular KCI
        pval_estimation = 'wild'
        n_points_wild_bootstrap = 1000
        # x -> x
        # y -> z
        # z -> y (conditioning on this)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        x = torch.tensor(Y_test, device=device).float()
        x_ho = torch.tensor(Y_train, device=device).float()
        z = torch.tensor(X_test, device=device).float()
        z_ho = torch.tensor(X_train, device=device).float()

        y = torch.tensor(Z_test, device=device).float()
        # y_ho = torch.tensor(Z_train, device=device).float()
        y_ho_x = torch.tensor(Z_train_y, device=device).float()
        y_ho_z = torch.tensor(Z_train_x, device=device).float()

        kernel_x = 'gaussian'
        kernel_z = 'kronecker'
        kernel_yz = 'gaussian'
        kernel_x_args = {'sigma2': x_ho.std().item() ** 2}
        kernel_yz_args = {'sigma2': y_ho_z.std().item() ** 2}
        kernel_z_args = {'sigma2': z_ho.std().item() ** 2}

        sigma_list = [0.5, 1.0, 2.0]
        param_dict_yx = {'sigma2': torch.tensor(sigma_list) * y_ho_x.std().item() ** 2}
        kernel_yx = 'gaussian'
        kernel_yx_args = {'sigma2': y_ho_x.std().item() ** 2}
        param_dict_yz = {'sigma2': torch.tensor(sigma_list) * y_ho_z.std().item() ** 2}

        print(measure)
        torch.cuda.empty_cache()
        gc.collect()
        if 'circe' in measure:
            kci = dependence_measures.CirceMeasure(kernel_x, kernel_z, kernel_yz,
                                                   kernel_yz_args, kernel_z_args, kernel_x_args,
                                                   biased=False)
        elif 'kci' in measure:
            kci = dependence_measures.KCIMeasure(kernel_x, kernel_yx, kernel_z, kernel_yz,
                                                 kernel_yx_args, kernel_x_args, kernel_yz_args, kernel_z_args,
                                                 biased=False)
        st = time.time()
        kci.find_regressors(x_ho, z_ho, y_z=y_ho_z, param_dict_yz=param_dict_yz, verbose=False,
                            cpu_solver=False, y_x=y_ho_x, param_dict_yx=param_dict_yx,
                            half_split_yx_estimator='xsplit' in measure or 'xzsplit' in measure,
                            half_split_yz_estimator='zsplit' in measure,
                            cpu_dtype=np.float128)
        print(f'regressors found in {time.time() - st}', flush=True)
        st = time.time()

        torch.cuda.empty_cache()
        gc.collect()

        kci_val, K, L = kci.compute_statistic(x, z, y, return_matrices=True)
        print(f'stat={kci_val} found in {time.time() - st}', flush=True)

        st = time.time()
        p_accepted_h0 = \
            kci.compute_pval(kci_val, pval_approx_type=pval_estimation, K=K, L=L, n_samples=n_points_wild_bootstrap,
                             chunk_size=1)
        print(f'pval found in {time.time() - st}', flush=True)

        del K, L

        pvals.append(p_accepted_h0)
        print(f'pval {p_accepted_h0}', flush=True)

    pvals = np.array(pvals)
    print(f'MEASURE: {measure}\nSTATES: {states}\nPVALS:\n{pvals}\n\n\n')


def main():
    run_test()


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
