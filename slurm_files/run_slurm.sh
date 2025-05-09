#!/bin/bash

module load anaconda/3
conda activate splitkci

mkdir -p $SCRATCH/splitkci_testing_results/$TASK

python $HOME/circe-testing-private/experiments/main_experiment.py \
  --config-file $HOME/circe-testing-private/configs/$CONFIG.yaml --problem_setup.task $TASK \
  --problem_setup.ca_kernels $KERNELS --problem_setup.measure $MEASURE --problem_setup.abc_holdout_sampling $XZY \
  --problem_setup.abc_budget $BUDGET --problem_setup.min_grid_points $MINN \
  --problem_setup.ground_truth $GROUND --files.filename "${TASK}_${PIND}_both_${PBOTH}_${MEASURE}_${KERNELS}_${XZY}_${BUDGET}_${MINN}_${GROUND}" \
  --problem_setup.dim $DIM --pval.n_holdout_resamples_start $SEEDS --pval.n_holdout_resamples_end $SEEDE \
  --problem_setup.train_test_split $TTSPLIT \
  --problem_setup.control $CONTROL --pval.split_alpha_ind $PIND --pval.split_alpha_both $PBOTH

echo "Job finished."
exit