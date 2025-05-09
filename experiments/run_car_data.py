# adapted from the RBPT paper https://github.com/felipemaiapolo/cit/tree/main (MIT license)
import os
import pandas as pd
import numpy as np
import torch
from splitkci import dependence_measures
import copy
import scipy
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import time
import gc

import warnings
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import OneOf


Section('params', 'save/load').params(
    save_folder=Param(str, default='test'),
    task=Param(OneOf(['simulated', 'true', 'both']), default='both'),
)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_data(data, n_vals, seed):
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
        rng = np.random.RandomState(seed)
        ind2 = rng.choice(np.sum(ind), np.sum(ind), replace=False)
        Y_ci[ind] = Y_ci[ind][ind2]
    return X, Y_ci, Z_bin


def get_loss(y, y_hat, loss='mae'):
    assert y.shape == y_hat.shape
    assert len(y.shape) == 2
    assert y.shape[1] == 1

    if loss == 'mae':
        return np.abs(y - y_hat)
    if loss == 'mse':
        return (y - y_hat) ** 2


class g:
    def __init__(self):
        pass

    def fit(self, X, Z, Y):
        if X is None:
            self.model = LinearRegression().fit(Z, Y)
        else:
            W = np.hstack((X, Z))
            self.model = LinearRegression().fit(W, Y)

    def predict(self, X, Z):
        if X is None:
            return self.model.predict(Z)
        else:
            W = np.hstack((X, Z))
            return self.model.predict(W)


def get_pval_gcm(X, Z, Y, g2, p_model):
    n = X.shape[0]
    rx = X-p_model.predict_proba(Z)[:,1].reshape(X.shape)
    ry = Y-g2.predict(None, Z)
    T = rx.squeeze()*ry.squeeze()
    pval = 2*(1 - scipy.stats.norm.cdf(abs(np.sqrt(n)*np.mean(T)/np.std(T))))
    return pval


def get_pval_rbpt(X, Z, Y, H, g1, loss='mse'):
    n = X.shape[0]
    XZ = np.hstack((X, Z))
    loss1 = get_loss(Y, g1.predict(X,Z).reshape((-1,1)), loss=loss)
    loss2 = get_loss(Y, H.reshape((-1,1)), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval


def get_pval_rbpt2(X, Z, Y, g1, h, loss='mae'):
    n = X.shape[0]
    XZ = np.hstack((X, Z))
    loss1 = get_loss(Y, g1.predict(X,Z).reshape((-1,1)), loss=loss)
    loss2 = get_loss(Y, h.predict(Z).reshape((-1,1)), loss=loss)
    T = loss2-loss1
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval


def get_pval_rbpt2_ub(X, Z, Y, g1, h, loss='mse'):
    assert loss == 'mse'
    n = X.shape[0]
    XZ = np.hstack((X, Z))
    g_pred = g1.predict(X,Z).reshape((-1,1))
    h_pred = h.predict(Z).reshape((-1,1))
    loss1 = get_loss(Y, g_pred, loss=loss)
    loss2 = get_loss(Y, h_pred, loss=loss)
    bias_correction = get_loss(g_pred, h_pred, loss=loss)
    T = loss2-loss1 + bias_correction
    pval = 1 - scipy.stats.norm.cdf(np.sqrt(n)*np.mean(T)/np.std(T))
    return pval


def get_h(X, y, validation_split=.1, verbose=False, random_state=None):

    ### Paramaters
    early_stopping_rounds=10
    loss='MultiRMSE'

    ### Validating
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state)

    m = CatBoostRegressor(loss_function = loss,
                          eval_metric = loss,
                          thread_count=-1,
                          random_seed=random_state)

    m.fit(X_train, y_train, verbose=verbose,
          eval_set=(X_val, y_val),
          early_stopping_rounds = early_stopping_rounds)


    ### Final model
    m2 = CatBoostRegressor(iterations=int(m.tree_count_),
                           loss_function = loss,
                           eval_metric = loss,
                           thread_count=-1,
                           random_seed=random_state)

    m2.fit(X, y, verbose=verbose)

    return m2


def run_non_kci_measures(pvals, s, cia, X_train, X_test, Y_train, Y_test, Z_train, Z_test):
    loss = 'mse'
    ###Fitting models
    g1 = g()
    g1.fit(X_train, Z_train, Y_train)
    g2 = g()
    g2.fit(None, Z_train, Y_train)
    h = get_h(Z_train, g1.predict(X_train, Z_train).squeeze())
    p = LogisticRegressionCV(cv=5, scoring='neg_log_loss', solver='liblinear', random_state=0).fit(Z_train,
                                                                                                   X_train.squeeze())
    H_test = np.sum(p.predict_proba(Z_test) * np.hstack((g1.predict(np.zeros(X_test.shape), Z_test).reshape(-1, 1),
                                                         g1.predict(np.ones(X_test.shape), Z_test).reshape(-1, 1))),
                    axis=1).reshape(-1, 1)
    pvals[s][cia]['rbpt'].append(get_pval_rbpt(X_test, Z_test, Y_test, H_test, g1, loss=loss))
    pvals[s][cia]['rbpt2'].append(get_pval_rbpt2(X_test, Z_test, Y_test, g1, h, loss=loss))
    pvals[s][cia]['rbpt2_ub'].append(get_pval_rbpt2_ub(X_test, Z_test, Y_test, g1, h, loss=loss))
    pvals[s][cia]['gcm'].append(get_pval_gcm(X_test, Z_test, Y_test, g2, p))


def run_kci_measures(pvals, s, cia, X_train, X_test, Y_train, Y_test, Zx_train, Zy_train, Z_test, chunk_size=500):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    A_train, A_test = torch.tensor(Y_train, device=device).float(), torch.tensor(Y_test, device=device).float()
    B_train, B_test = torch.tensor(X_train, device=device).float(), torch.tensor(X_test, device=device).float()
    Ca_train, C_test = torch.tensor(Zy_train, device=device).float(), torch.tensor(Z_test, device=device).float()
    Cb_train = torch.tensor(Zx_train, device=device).float()

    kernel_a = 'gaussian'
    kernel_b = 'kronecker'  # X is binary
    kernel_ca = 'gaussian'
    kernel_cb = 'gaussian'

    kernel_a_args = {'sigma2': A_train.std().item() ** 2}
    kernel_b_args = {'sigma2': B_train.std().item() ** 2}

    kernel_ca_args = {'sigma2': Ca_train.std().item() ** 2}
    param_dict_ca = {'sigma2': torch.tensor([0.5, 1.0, 2.0]) * Ca_train.std().item() ** 2}

    kernel_cb_args = {'sigma2': Cb_train.std().item() ** 2}
    param_dict_cb = {'sigma2': torch.tensor([0.5, 1.0, 2.0]) * Cb_train.std().item() ** 2}

    # kci, kci_a, kci_ab, circe
    # only need to compute kci and kci_ab
    kci = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb, kernel_ca_args, kernel_a_args,
                                         kernel_cb_args, kernel_b_args, biased=False,
                                         half_split_ca_estimator=False, half_split_cb_estimator=False,
                                         circe_a=False, circe_b=False)
    kci_ab = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb, kernel_ca_args, kernel_a_args,
                                            kernel_cb_args, kernel_b_args, biased=False,
                                            half_split_ca_estimator=True, half_split_cb_estimator=True,
                                            circe_a=False, circe_b=False)
    kci_a = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb, kernel_ca_args, kernel_a_args,
                                           kernel_cb_args, kernel_b_args, biased=False,
                                           half_split_ca_estimator=True, half_split_cb_estimator=False,
                                           circe_a=False, circe_b=False)
    circe = dependence_measures.KCIMeasure(kernel_a, kernel_ca, kernel_b, kernel_cb, kernel_ca_args, kernel_a_args,
                                           kernel_cb_args, kernel_b_args, biased=False,
                                           half_split_ca_estimator=False, half_split_cb_estimator=False,
                                           circe_a=True, circe_b=False)

    kci.find_regressors(A_train, B_train, Cb_train, c_a=Ca_train,
                        param_dict_ca=param_dict_ca, param_dict_cb=param_dict_cb)
    kci_ab.find_regressors(A_train, B_train, Cb_train, c_a=Ca_train,
                           param_dict_ca=param_dict_ca, param_dict_cb=param_dict_cb)
    kci_a.krr_ca = kci_ab.krr_ca
    kci_a.krr_cb = kci.krr_cb
    kci_a._regression_done = True
    circe.krr_cb = kci.krr_cb
    circe._regression_done = True

    for measure in ['kci', 'kci_a', 'kci_ab', 'circe']:
        torch.cuda.empty_cache()
        gc.collect()

        print(f'...running {measure}', flush=True)
        kci_val, K, L = eval(f'{measure}.compute_statistic(A_test, B_test, C_test, return_matrices=True)')
        pvals[s][cia][measure].append(
            eval(f'{measure}.compute_pval(kci_val, K=K, L=L, pval_approx_type=\'wild\', n_samples=1000, chunk_size=chunk_size)'))


@param('params.save_folder')
@param('params.task')
def run_test(save_folder, task):
    n_vals = 20
    test_ratio = 0.3
    states = ['ca', 'il', 'mo', 'tx']
    ci_measures = ['gcm', 'rbpt', 'rbpt2', 'rbpt2_ub', 'kci', 'kci_a', 'kci_ab', 'circe']

    companies = dict()
    for s in states:
        data = pd.read_csv(os.path.join(save_folder, s + '-per-zip.csv'))
        companies[s] = list(set(data.companies_name))

    if task in ['simulated', 'both']:
        print('Running simulated', flush=True)
        pvals_simulated = {s: {c: {j: list() for j in ci_measures} for c in companies[s]} for s in states}

        for s in states:
            for cia in companies[s]:
                data = pd.read_csv(os.path.join(save_folder, s + '-per-zip.csv'))
                data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
                data = data.loc[data.companies_name == cia]
                # simulated H0
                for seed in range(50):
                    print(f'Seed {seed}', flush=True)

                    X, Y_ci, Z_bin = get_data(data, n_vals, seed)
                    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
                        X, Y_ci, Z_bin, test_size=test_ratio, random_state=seed)

                    start_t = time.time()
                    run_non_kci_measures(pvals_simulated, s, cia, X_train, X_test, Y_train, Y_test, Z_train, Z_test)
                    print(f'...non-kci done in {time.time() - start_t}', flush=True)

                    start_t = time.time()
                    run_kci_measures(pvals_simulated, s, cia, X_train, X_test, Y_train, Y_test,
                                     Z_train, Z_train, Z_test)
                    print(f'...kci done in {time.time() - start_t}', flush=True)

        torch.save(pvals_simulated, os.path.join(save_folder, 'car_pvals_simulated.pt'))

    if task in ['true', 'both']:
        print('Running true', flush=True)
        pvals_true = {s: {'all': {j: list() for j in ci_measures}} for s in states}
        for s in states:
            data = pd.read_csv(os.path.join(save_folder, s + '-per-zip.csv'))
            data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
            # actual test
            Z = np.array(data.state_risk).reshape((-1, 1))
            Y = np.array(data.combined_premium).reshape((-1, 1))
            X = (1 * np.array(data.minority)).reshape((-1, 1))

            X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=test_ratio,
                                                                                 random_state=42)
            start_t = time.time()
            run_non_kci_measures(pvals_true, s, 'all', X_train, X_test, Y_train, Y_test, Z_train, Z_test)
            print(f'...non-kci done in {time.time() - start_t}', flush=True)

            # using fewer points due to memory limits
            X_train, _, _, Y_train, Z_train_x, Z_train_y = train_test_split(X_train, Y_train, Z_train,
                                                                            test_size=0.5,
                                                                            random_state=2 * 42)
            start_t = time.time()
            run_kci_measures(pvals_true, s, 'all', X_train, X_test, Y_train, Y_test,
                             Z_train_x, Z_train_y, Z_test, chunk_size=10)
            print(f'...kci done in {time.time() - start_t}', flush=True)

        torch.save(pvals_true, os.path.join(save_folder, 'car_pvals_true.pt'))


def main():
    warnings.filterwarnings("ignore")
    run_test()


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Insurance data')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
