# Testing with kernel-based measures of conditional independence

Code for 

"Practical Kernel Tests of Conditional Independence"

Roman Pogodin, Antonin Schrab, Yazhe Li, Danica J. Sutherland, Arthur Gretton

https://arxiv.org/abs/2402.13196
 
## Installation

The slurm scripts assume there's a conda environment `splitkci`.
In addition to `python 3.9, pytorch >=2.0`, this repository requires 
```
pip install fastargs ratinabox>=1.15.3
```

The main methods are packaged into a `splitkci` package. To install, clone this repository and run
```
pip install -e .
```

## Experiments

The code assumes you have `$SCRATCH` and `$HOME` folders, and run the slurm scripts from `./slurm_files`.
It's recommended to run everything on a GPU (the code will use it if it's available). The experiments require at most 24 GB of RAM and a similar amount of VRAM, 
but the VRAM requirement can be lowered by using a small `chunk_size` in wild bootstrap p-value computation (e.g. `ci_test.test(..., chunk_size=100)` in the code in `./experiments`).

Plots for all experiments are made in './notebooks'.

#### RatInABox

Data for these experiments is generated using `./ratinabox_data/run_slurm_rats_gen.sh`. See the readme in `./ratinabox_data`. 

The scripts for these experiments are
`start_experiments_rats_budget.sh` (budget experiments) and `start_experiments_rats.sh` (all comparisons).

#### Toy task

The script for this experiment is `start_experiments_randn.sh` (including the p-value approximation comparisons).

### Car insurance data

This code follows https://github.com/felipemaiapolo/cit.git and re-uses some of their code (see comments in the individual files).

The script is `run_slurm_cards.sh` and can be run as
```
sbatch --gres=gpu:ampere:1 -c 8 --mem=48G -t 12:00:00 --partition=main,long --output="./logs/car_insurance.out" \
          --export=ALL,TASK='simulated' ./run_slurm_cars.sh
```
for the simulated null and
```sbatch --gres=gpu:ampere:1 -c 8 --mem=48G -t 12:00:00 --partition=main,long --output="./logs/car_insurance_true.out" \
          --export=ALL,TASK='true' ./run_slurm_cars.sh
```
for the true p-values.

The script will download the dataset from `https://github.com/felipemaiapolo/cit.git`.
