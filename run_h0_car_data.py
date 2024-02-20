# adapted from the RBPT paper https://github.com/felipemaiapolo/cit/tree/main
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import time
import torch
import dependence_measures

import warnings


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def exp2_kci(it, n_vals, loss, alpha, B):
    states = ['ca', 'il', 'mo', 'tx']
    ci_measures = ['kci', 'kci_xsplit', 'circe']  # ['kci', 'circe'] # , 'kci_xsplit'
    pvals = {}
    times = {}

    for state in states:
        pvals[state] = {}
        times[state] = {}
        for measure in ci_measures:
            pvals[state][measure] = []
            times[state][measure] = []

    count = 0

    for s in states:

        # assumes a cloned RBPT repo in $HOME
        data = pd.read_csv(os.path.join(os.environ['HOME'],
                                        'cit/data/car-insurance-public/data/' + s + '-per-zip.csv'))
        companies = list(set(data.companies_name))

        for cia in companies:

            data = pd.read_csv(os.path.join(os.environ['HOME'],
                                            'cit/data/car-insurance-public/data/' + s + '-per-zip.csv'))
            data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
            data = data.loc[data.companies_name == cia]

            Z = np.array(data.state_risk).reshape((-1, 1))
            Y = np.array(data.combined_premium).reshape((-1, 1))
            X = (1 * np.array(data.minority)).reshape((-1, 1))

            # bins = np.percentile(Z, np.linspace(0,100,n_vals+2))
            bins = np.linspace(np.min(Z), np.max(Z), n_vals + 2)
            bins = bins[1:-1]
            Y_ci = copy.deepcopy(Y)
            Z_bin = np.array([find_nearest(bins, z) for z in Z.squeeze()]).reshape(Z.shape)

            for val in np.unique(Z_bin):
                ind = Z_bin == val
                rng = np.random.RandomState(it)
                ind2 = rng.choice(np.sum(ind), np.sum(ind), replace=False)
                Y_ci[ind] = Y_ci[ind][ind2]

            X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y_ci, Z_bin, test_size=.3,
                                                                                 random_state=it)

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
            y_ho = torch.tensor(Z_train, device=device).float()
            y_ho_x = y_ho.clone()
            y_ho_z = y_ho.clone()

            kernel_x = 'gaussian'  # 'kronecker'
            kernel_z = 'kronecker'  # 'gaussian'
            kernel_yz = 'gaussian'
            kernel_x_args = {'sigma2': x_ho.std().item() ** 2}
            kernel_yz_args = {'sigma2': y_ho.std().item() ** 2}
            kernel_z_args = {'sigma2': z_ho.std().item() ** 2}

            #             if yx_kernels == 'gaussian':
            param_dict_yx = {'sigma2': torch.tensor([0.5, 1.0, 2.0]) * y_ho.std().item() ** 2}
            kernel_yx = 'gaussian'

            kernel_yx_args = {'sigma2': 1.0, 'n': 3}
            param_dict_yz = {'sigma2': torch.tensor([0.5, 1.0, 2.0]) * y_ho.std().item() ** 2}

            for measure in ci_measures:
                print(measure, flush=True)
                if 'circe' in measure:
                    kci = dependence_measures.CirceMeasure(kernel_x, kernel_z, kernel_yz,
                                                           kernel_yz_args, kernel_z_args, kernel_x_args,
                                                           biased=False)
                elif 'kci' in measure:
                    kci = dependence_measures.KCIMeasure(kernel_x, kernel_yx, kernel_z, kernel_yz,
                                                         kernel_yx_args, kernel_x_args, kernel_yz_args, kernel_z_args,
                                                         biased=False)
                kci.find_regressors(x_ho, z_ho, y_z=y_ho_z, param_dict_yz=param_dict_yz, verbose=False,
                                    cpu_solver=True, y_x=y_ho_x, param_dict_yx=param_dict_yx,
                                    half_split_yx_estimator='xsplit' in measure or 'xzsplit' in measure,
                                    half_split_yz_estimator='zsplit' in measure, cpu_dtype=np.float128)
                kci_val, K, L = kci.compute_statistic(x, z, y, return_matrices=True)
                print(kci_val, flush=True)

                p_accepted_h0 = \
                    kci.compute_pval(kci_val, pval_approx_type=pval_estimation, K=K, L=L,
                                     n_samples=n_points_wild_bootstrap)
                print(p_accepted_h0, flush=True)

                pvals[s][measure].append([count, p_accepted_h0])

        count += 1

    return pvals


def run_test():
    alpha = .05
    loss = 'mse'
    B = 100
    n_vals = 20

    start_time = time.time()
    out = list()
    for it in range(50):
        out.append(exp2_kci(it, n_vals, loss, alpha, B))
    print(time.time() - start_time)

    torch.save(out, 'car-final.pt')


def main():
    warnings.filterwarnings("ignore")
    run_test()


if __name__ == "__main__":
    main()
