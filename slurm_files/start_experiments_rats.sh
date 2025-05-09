#!/bin/bash

TASK='ratinabox'
DIM=100
PBOTH=0.05
PIND=0.05
KERNELS='all'

for GROUND in 'H0' 'H1'; do
  for MEASURE in 'gcm' 'rbpt2_ub' 'circe' 'kci' 'kci_asplit' 'kci_absplit'; do
    XZY='joint'
      MINN=200
      BUDGET=0

      CONTROL='False'
        for TTSPLIT in 'False' 0.5 100; do
          sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/refactored_${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
        done

      CONTROL='True'
        TTSPLIT='False'
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/refactored_${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh

    XZY='separate'
      MINN=100
      BUDGET=200
      CONTROL='False'
      for TTSPLIT in 'False' 100; do
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/refactored_${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
      done

    XZY='separate'
      MINN=200
      BUDGET=400
      CONTROL='False'
      for TTSPLIT in 'False' 100 200; do
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/refactored_${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
      done
  done

  MEASURE='kci_absplit'
    XZY='joint'
    MINN=200
    BUDGET=0

    CONTROL='False'
    TTSPLIT='auto'
      sbatch --gres=gpu:1 -c 4 --mem=24G -t 24:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh

  XZY='separate'
    MINN=100
    BUDGET=200
    CONTROL='False'
    TTSPLIT='auto'
      sbatch --gres=gpu:1 -c 4 --mem=24G -t 24:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh

  XZY='separate'
    MINN=200
    BUDGET=400
    CONTROL='False'
    TTSPLIT='auto'
      sbatch --gres=gpu:1 -c 4 --mem=24G -t 24:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
done


# RBPT2 (biased) only

TASK='ratinabox'
DIM=100
PBOTH=0.05
PIND=0.05
KERNELS='all'

for GROUND in 'H0' 'H1'; do
  for MEASURE in 'rbpt2'; do
    XZY='joint'
      MINN=200
      BUDGET=0

      CONTROL='False'
        for TTSPLIT in 'False' 0.5 100; do
          sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
        done

      CONTROL='True'
        TTSPLIT='False'
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh

    XZY='separate'
      MINN=100
      BUDGET=200
      CONTROL='False'
      for TTSPLIT in 'False' 100; do
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
      done

    XZY='separate'
      MINN=200
      BUDGET=400
      CONTROL='False'
      for TTSPLIT in 'False' 100 200; do
        sbatch --gres=gpu:1 -c 4 --mem=24G -t 12:00:00 --partition=main,long --output="./logs/${TASK}_${MEASURE}_${KERNELS}_${XZY}_${MINN}_${BUDGET}_${GROUND}_${PIND}_${PBOTH}_${TTSPLIT}_${CONTROL}.out" \
          --export=ALL,CONFIG='default_config_rat',MEASURE=$MEASURE,TASK=$TASK,KERNELS=$KERNELS,XZY=$XZY,MINN=$MINN,BUDGET=$BUDGET,GROUND=$GROUND,DIM=$DIM,TTSPLIT=$TTSPLIT,CONTROL=$CONTROL,SEEDS=0,SEEDE=100,PIND=$PIND,PBOTH=$PBOTH \
          ./run_slurm.sh
      done
  done
done

