import gym

from network_snn import SpikingNN
from ppo_snn import PPO
from hyper_params import HyperParams


def train(env, params):
    print('Training ...')

    # Create a model for PPO
    model = PPO(policy_class=SpikingNN, env=env, params=params)

    # Train the PPO model with a specified total timesteps
    model.learn(total_timesteps=100000)


if __name__ == '__main__':
    hyper_params = HyperParams()

    # neuron parameters
    hyper_params.beta = 0.99

    # architecture parameters
    hyper_params.n_hidden = [64, 64]

    # training parameters
    hyper_params.lr = 0.005  # learning rate
    hyper_params.gamma = 0.95   # discount factor for rewards
    hyper_params.timesteps_per_batch = 4800  # timesteps per batch
    hyper_params.max_timesteps_per_episode = 1600  # timesteps per episode
    hyper_params.ppo_action_stddev = 0.5  # standard deviation for sampling actions from the multivariate normal distribution
    hyper_params.ppo_n_updates_per_batch = 5  # number of times actor and critic networks are updated for a single batch of observations
    hyper_params.ppo_clip = 0.2  # clip the ratio to [1 - clip, 1 + clip]

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)