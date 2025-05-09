#!/bin/bash
DIM=100

for BUDGET in 200 500 1000; do
  for GROUND in 'H0' 'H1'; do
    for MEASURE in 'kci_absplit' 'kci' 'kci_asplit'; do
      for KERNELS in 'all'; do
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 02:00:00 --partition=main,long --output="./logs/budget_${TASK}_${MEASURE}_${KERNELS}_${BUDGET}_${GROUND}" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,KERNELS=$KERNELS,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,SEEDS=0,SEEDE=100 \
          ./run_slurm_rats_budget.sh
      done
    done
  done
done
