# Testing with kernel-based measures of conditional independence

Dependencies: `python3.9, pytorch 2.0, fastargs`. It's recommended to run everything on a GPU (the code will use it if it's available).

The code saves everything in the `$HOME` directory.

For the car data, it assumes there's a folder `$HOME/cit/data/car-insurance-public`.
Run `git clone https://github.com/felipemaiapolo/cit.git` in `$HOME` to create it (since the data is distributed with the RBPT code).

For the plotting notebooks, we heavily re-use the code from `https://github.com/felipemaiapolo/cit.git` [1] to reproduce their experiments.

### Artificial data experiments

``` 
python ./toy_data_experiment.py \
  --config-file ./configs/$CONFIG.yaml \
  --problem_setup.yx_kernels $KERNELS --problem_setup.measure $MEASURE --problem_setup.xzy_holdout_sampling $XZY \
  --problem_setup.n_points $NPOINTS --problem_setup.n_xy_points $NPOINTS  --problem_setup.min_n_zy_points 100 \
  --problem_setup.task $TASK --problem_setup.ground_truth $GROUND --files.filename "${TASK}_${MEASURE}_${KERNELS}_${XZY}_${NPOINTS}_${NPOINTSZY}_${GROUND}" \
  --problem_setup.max_n_zy_points $MAXZY 
```

Task 1-3: `$CONFIG='default_config_2dim'` (`default_config_gamma` for gamma approximation + change `--files.filename` to save to a different file)

Task 4: `$CONFIG='rbpt_task_experiments'`

All tasks: `$KERNELS='gaussian'` or `'all'` (for best kernel for the first regression)

`$MEASURE='kci'` or `kci_xsplit`, `circe`, `gcm`, `rbpt2`, `rbpt2_ub` (unbiased version)

`$XZY='joint'` for standard data regime or `separate` for the unbalanced data regime.

`$NPOINTS=100` or 200 (double that for CIRCE)

`$TASK=get_xzy_randn` (task 1), `get_xzy_randn_nl` (task 2), `get_xzy_circ` (task 3), `get_xzy_rbpt` (task 4)

`$GROUND='H0'` or `H1`

`$MAXZY=1000` for n=100 and 2000 for n=200 (not doubled for CIRCE!)

Task 4: 

For `H0`, add `--problem_setup.rbpt_c 0.0 --problem_setup.rbpt_gamma 0.02 --problem_setup.rbpt_seed $SEED` 

For `H1`, add `--problem_setup.rbpt_c 0.1 --problem_setup.rbpt_gamma 0.0 --problem_setup.rbpt_seed $SEED` 

for `$SEED` in 1 2 3 4 5


### Car insurance data

Simulated null:
`python ./run_h0_car_data.py`

Actual p-values: `python ./run_pval_car_data.py --test.measure $MEASURE` 
for `$MEASURE='kci'` or `kci_xsplit` or `circe`. 

RBPT2 and GCM are run as in https://github.com/felipemaiapolo/cit.git

For RBPT2' (unbiased version), replace `get_pval_rbpt2` in https://github.com/felipemaiapolo/cit/blob/main/exp2.py with

```python
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
```

[1] Maia Polo, Felipe, Yuekai Sun, and Moulinath Banerjee. "Conditional independence testing under misspecified inductive biases." Advances in Neural Information Processing Systems 36 (2024).
