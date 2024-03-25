import gym

from network_snn import SpikingNN
from ppo_snn import PPO_SNN


def train(env, params):
    print('Training ...')

    model = PPO_SNN(policy_class=SpikingNN, env=env, params=params)


if __name__ == '__main__':
    hyper_params = {}

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)