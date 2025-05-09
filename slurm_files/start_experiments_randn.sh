#!/bin/bash

TASK='randn'
DIM=2
PBOTH=0.05
PIND=0.05
KERNELS='gaussian'
CONTROL='False'
XZY='joint'
TTSPLIT=100
MINN=200
BUDGET=0

for GROUND in 'H0'; do
  for MEASURE in 'rbpt2' 'rbpt2_ub' 'circe' 'kci' 'kci_asplit' 'kci_absplit'; do
    sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}_wild.out" \
    --export=ALL,CONFIG='default_config_randn',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
    ./run_slurm.sh
  done

  for MEASURE in 'circe' 'kci' 'kci_asplit' 'kci_absplit'; do
    sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}_gamma.out" \
    --export=ALL,CONFIG='default_config_gamma',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
    ./run_slurm.sh
  done
done


