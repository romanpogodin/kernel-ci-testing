This data generation code requires https://github.com/RatInABox-Lab/RatInABox/ (>=1.15.3) installed.

This codes simulates a rat running in a maze, and "records" from its hippocampal neurons.
Here, we work with head direction cells (each neuron has a preferred head direction,
and is only active when the rat looks in that direction) and grid cells
(each neuron has a grid of preferred locations in the maze, and only fires
when the rat is there). We also look and conjunctive cells, which implement
the AND operation for head direction and grid cells. 

To generate the data, for each seed we let the rat run for several minutes. 
Then, we subsample the recordings such that the points are approximately iid. 
The sampling rate is defined by the time constants in the noise processes.

Each seed take a few minutes, so you can run it as several jobs:
```
NOISE='0.1'
NCELLS=100
for MINSEED in `seq 0 5 95`; do
MAXSEED=$((MINSEED + 5))
sbatch -c 1 --mem=4G -t 3:00:00 --partition=long-cpu --output="./rats_${MINSEED}.out" --export=ALL,MINSEED=$MINSEED,MAXSEED=$MAXSEED,NOISE=$NOISE,NCELLS=$NCELLS  ./run_slurm_rats_gen.sh
done
```
As an aside, more noise leads to an easier problem 
(since dependence/independence comes from noise). For 0.1 like above, 
the problem is sufficiently hard to see difference between methods/dataset sizes.

Finally, we only use data generation to make cell populations (e.g. without nonlinearities).
The final data loading is done in the `load_rat_data` function (in `experiments/tasks.py`).

There, we construct A as a head direction population, 
B as position+head direction population,
and C as another, independent position+head direction population.

Under H0, A/B/C are connected through the actual head direction and position.
Under H1, B gets the head direction from A.