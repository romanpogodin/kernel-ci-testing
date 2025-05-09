#!/bin/bash

module load anaconda/3
conda activate splitkci

mkdir $SCRATCH/ratinabox_saved_data

python $HOME/circe-testing-private/ratinabox_data/ratinabox_data_generation.py \
  --params.path $SCRATCH/ratinabox_saved_data --params.n_cells $NCELLS  --params.n_points 3000 \
  --params.min_seed $MINSEED --params.max_seed $MAXSEED  --params.noise_std $NOISE


#NOISE='0.1'
#NCELLS=100
#for MINSEED in `seq 0 5 95`; do
#MAXSEED=$((MINSEED + 5))
#sbatch -c 1 --mem=4G -t 3:00:00 --partition=long-cpu --output="./rats_${MINSEED}.out" --export=ALL,MINSEED=$MINSEED,MAXSEED=$MAXSEED,NOISE=$NOISE,NCELLS=$NCELLS  ./run_slurm_rats_gen.sh
#done

echo "Job finished."
exit