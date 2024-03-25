import gym

from network_snn import SpikingNN
from ppo_snn import PPO


def train(env, params):
    print('Training ...')

    model = PPO(policy_class=SpikingNN, env=env, params=params)


if __name__ == '__main__':
    hyper_params = {}

    # neuron parameters
    hyper_params['beta'] = 0.99

    # architecture parameters
    hyper_params['n_hidden'] = [64, 64]

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)