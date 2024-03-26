import gym

from network_snn import SpikingNN
from ppo_snn import PPO
from hyper_params import HyperParams


def train(env, params):
    print('Training ...')

    model = PPO(policy_class=SpikingNN, env=env, params=params)


if __name__ == '__main__':
    hyper_params = HyperParams()

    # neuron parameters
    hyper_params.beta = 0.99

    # architecture parameters
    hyper_params.n_hidden = [64, 64]

    # training parameters
    hyper_params.timesteps_per_batch = 4800  # timesteps per batch
    hyper_params.max_timesteps_per_episode = 1600  # timesteps per episode

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)