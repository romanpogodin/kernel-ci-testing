#!/bin/bash

module load anaconda/3
conda activate splitkci

mkdir -p $SCRATCH/splitkci_testing_results/car_insurance

wget -P $SCRATCH/splitkci_testing_results/car_insurance https://raw.githubusercontent.com/felipemaiapolo/cit/refs/heads/main/data/car-insurance-public/data/ca-per-zip.csv
wget -P $SCRATCH/splitkci_testing_results/car_insurance https://raw.githubusercontent.com/felipemaiapolo/cit/refs/heads/main/data/car-insurance-public/data/il-per-zip.csv
wget -P $SCRATCH/splitkci_testing_results/car_insurance https://raw.githubusercontent.com/felipemaiapolo/cit/refs/heads/main/data/car-insurance-public/data/mo-per-zip.csv
wget -P $SCRATCH/splitkci_testing_results/car_insurance https://raw.githubusercontent.com/felipemaiapolo/cit/refs/heads/main/data/car-insurance-public/data/tx-per-zip.csv

python $HOME/circe-testing-private/experiments/run_car_data.py \
  --params.save_folder $SCRATCH/splitkci_testing_results/car_insurance --params.task $TASK

echo "Job finished."
exit

#sbatch --gres=gpu:ampere:1 -c 8 --mem=48G -t 12:00:00 --partition=main,long --output="./logs/car_insurance.out" \
#          --export=ALL,TASK='simulated' ./run_slurm_cars.sh

#sbatch --gres=gpu:ampere:1 -c 8 --mem=48G -t 12:00:00 --partition=main,long --output="./logs/car_insurance_true.out" \
#          --export=ALL,TASK='true' ./run_slurm_cars.sh