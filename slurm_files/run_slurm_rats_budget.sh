#!/bin/bash

module load anaconda/3
conda activate splitkci

mkdir -p $SCRATCH/splitkci_testing_results/budget_experiment


python $HOME/circe-testing-private/experiments/rat_data_split_study.py \
  --config-file $HOME/circe-testing-private/configs/$CONFIG.yaml \
  --problem_setup.ca_kernels $KERNELS --problem_setup.measure $MEASURE \
  --problem_setup.ground_truth $GROUND --files.filename "budget_${MEASURE}_${KERNELS}_${BUDGET}_${GROUND}" \
  --problem_setup.dim $DIM --pval.n_holdout_resamples_start $SEEDS --pval.n_holdout_resamples_end $SEEDE \
  --problem_setup.budget $BUDGET


#sbatch --gres=gpu:1 -c 4 --mem=24G -t 1-00:00 --partition=main --output=./randn.out --export=CONFIG='kci_randn_h0'  ./run_slurm.sh

#pyenv deactivate
echo "Job finished."
exit