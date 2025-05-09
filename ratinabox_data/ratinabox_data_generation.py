from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
import os


from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section


Section('params', 'Data generation params').params(
    min_seed=Param(int, default=0),
    max_seed=Param(int, default=1),
    n_cells=Param(int, default=100),
    n_points=Param(int, default=3000),
    path=Param(str, default='./'),
    noise_std=Param(float, default=0.1),
)


class Rat:
    def __init__(self, n_cells, noise_std=0.1):
        self.n_cells = n_cells

        self.env = Environment(params={"aspect": 1, "scale": 1})
        self.env.add_wall([[0.25, 0.25], [0.25, 0.75]])
        self.env.add_wall([[0.25, 0.75], [0.75, 0.75]])
        self.env.add_wall([[0.75, 0.75], [0.75, 0.25]])
        self.env.add_wall([[0.75, 0.25], [0.25, 0.25]])

        self.agent = Agent(self.env, {'rotational_velocity_coherence_time': 0.08, 'speed_coherence_time': 0.08})
        self.agent.pos = np.array([0.0, 0.0])
        self.agent.speed_mean = 0.2

        self.noise_std = noise_std
        self.noise_coherence_time = 0.08  # autocorr of OU process is sigma*2 * exp(-|t-t'| / this var), so 2 / self.noise_coherence_time is good for low corr

        self.grid_cells = GridCells(self.agent, params={
            'n': self.n_cells,
            'gridscale': np.linspace(0.2, 0.6, self.n_cells),
            'name': 'grid',
            "noise_std": self.noise_std,  # 0 means no noise, std of the noise you want to add (Hz)
            "noise_coherence_time": self.noise_coherence_time,
            'color': 'C1'})
        self.head_dir_cells = HeadDirectionCells(self.agent, params={
            'n': self.n_cells,
            'name': 'head_dir',
            "noise_std": self.noise_std,  # 0 means no noise, std of the noise you want to add (Hz)
            "noise_coherence_time": self.noise_coherence_time,
            'color': 'C5'})

        self.grid_cells_ind = GridCells(self.agent, params={
            'n': self.n_cells,
            'gridscale': np.linspace(0.2, 0.6, self.n_cells),
            'name': 'grid_ind',
            "noise_std": self.noise_std,  # 0 means no noise, std of the noise you want to add (Hz)
            "noise_coherence_time": self.noise_coherence_time,
            'color': 'C1'})
        self.head_dir_cells_ind = HeadDirectionCells(self.agent, params={
            'n': self.n_cells,
            'name': 'head_dir_ind',
            "noise_std": self.noise_std,  # 0 means no noise, std of the noise you want to add (Hz)
            "noise_coherence_time": self.noise_coherence_time,
            'color': 'C5'})

    def _update(self, dt):
        self.agent.update(dt=dt)
        self.grid_cells.update()
        self.head_dir_cells.update()
        self.grid_cells_ind.update()
        self.head_dir_cells_ind.update()

    def get_history(self, time_indices):
        full_history = dict()
        ag_history = self.agent.get_history_arrays()
        full_history['pos'] = ag_history['pos'][time_indices]
        full_history['head_direction'] = ag_history['head_direction'][time_indices]

        for cells in [self.grid_cells, self.head_dir_cells, self.grid_cells_ind, self.head_dir_cells_ind]:
            full_history[cells.name + '_rate'] = cells.get_history_arrays()['firingrate'][time_indices]

        return full_history

    def simulate(self, max_time, dt):
        for _ in range(int(max_time / dt)):
            self._update(dt)

    def get_n_iid_points(self, n, dt, warm_up_n=100):
        slowest_time_const = max(self.noise_coherence_time, self.agent.speed_coherence_time,
                                 self.agent.rotational_velocity_coherence_time)
        max_time = 10 * (warm_up_n + n) * slowest_time_const
        sampling_dt = int(10 * slowest_time_const / dt)
        self.simulate(max_time, dt)

        time_indices = slice(warm_up_n * sampling_dt,
                             (warm_up_n + n) * sampling_dt,
                             sampling_dt)
        full_history = self.get_history(time_indices)

        return full_history


@param('params.path')
@param('params.n_cells')
@param('params.n_points')
@param('params.noise_std')
def save_rats(seeds, path, n_cells, n_points, noise_std):
    for seed in seeds:
        np.random.seed(seed)
        rat = Rat(n_cells, noise_std)
        full_history = rat.get_n_iid_points(n_points, dt=50e-3, warm_up_n=100)
        name = f'{n_cells}_cells_{n_points}_points_noise_{noise_std}_seed_{seed}.npy'
        np.save(os.path.join(path, name), full_history)


@param('params.min_seed')
@param('params.max_seed')
def main(min_seed, max_seed):
    save_rats(np.arange(min_seed, max_seed))


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Data generation')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
