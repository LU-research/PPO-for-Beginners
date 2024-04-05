import gym
import torch

from network_snn import SpikingNN
from ppo_snn import PPO
from utils import HyperParams


def drop_mem_states(model, saved_model):
    # get keys for saved membrane potentials from the state_dict
    removed_keys = []
    for key in model.state_dict():
        if key.endswith('mem'):
            removed_keys += [key]

    # remove membrane potentials from the state_dict
    for key in removed_keys:
        del saved_model[key]

    return saved_model


def train(env, params):
    # Create a model for PPO
    model = PPO(policy_class=SpikingNN, env=env, params=params)

    # load saved model
    if params.load_last_model is True:
        saved_actor = torch.load(params.actor_model)
        saved_critic = torch.load(params.critic_model)

        # drop saved membrane potentials from the saved model
        saved_actor = drop_mem_states(model.actor, saved_actor)
        saved_critic = drop_mem_states(model.critic, saved_critic)

        model.actor.load_state_dict(saved_actor, strict=False)
        model.critic.load_state_dict(saved_critic, strict=False)

    # Train the PPO model with a specified total timesteps
    model.learn(total_timesteps=200_000_000)


if __name__ == '__main__':
    hyper_params = HyperParams()

    # neuron parameters
    hyper_params.beta = 0.99

    # spike encoding and decoding parameters
    hyper_params.T = 100  # number of time steps for simulation per sample

    # architecture parameters
    hyper_params.n_hidden = [64, 64]

    # training parameters
    hyper_params.ppo_render = True  # render actions taken by the actor in the first episode in rollout
    hyper_params.lr = 0.005  # learning rate
    hyper_params.gamma = 0.95   # discount factor for rewards
    hyper_params.timesteps_per_batch = 2048  # timesteps per batch
    hyper_params.max_timesteps_per_episode = 200  # timesteps per episode
    hyper_params.ppo_action_stddev = 0.5  # standard deviation for sampling actions from the multivariate normal distribution
    hyper_params.ppo_n_updates_per_batch = 5  # number of times actor and critic networks are updated for a single batch of observations
    hyper_params.ppo_clip = 0.2  # clip the ratio to [1 - clip, 1 + clip]
    
    # saving and loading models
    hyper_params.actor_model = './snn_ppo_actor.pth'
    hyper_params.critic_model = './snn_ppo_critic.pth'
    hyper_params.save_freq = 10  # How often to save the model
    hyper_params.load_last_model = True  # whether last saved model should be loaded

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)

    # close the environment
    pendulum_env.close()