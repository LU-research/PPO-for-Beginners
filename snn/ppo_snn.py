import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from utils import Logger


# Following line is there for a torch bug
torch.pi = math.pi


class PPO:
    def __init__(self, policy_class, env, params) -> None:
        # Extract environment information
        self.env = env
        self.params = params
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(params.beta, params.T, self.obs_dim, params.n_hidden, self.act_dim)
        self.critic = policy_class(params.beta, params.T, self.obs_dim, params.n_hidden, 1)

        # create the covariance matrix for sampling actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.params.ppo_action_stddev)
        self.cov_mat = torch.diag(self.cov_var)

        # create the optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.params.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.params.lr)

        # logger to track variations in different quantities
        self.logger = Logger()
        self.logger.actor_loss = []

    def get_action(self, obs):
        # convert obs to torch tensor
        obs = torch.tensor(obs, dtype=torch.float)

        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        lif1_spk_rec, lif2_spk_rec, lif3_mem_rec = self.actor(obs)
        # potential at last time step is considered as mean of the Multivariate Normal Distribution
        mean = lif3_mem_rec[-1]

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob are tensors with computation graphs, so I want to get rid of the graph and just convert the action to numpy array. log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_reward):
        batch_reward_to_go = []

        # Iterate starting from last episode to maintain same order in batch_rewards_to_go
        for eps_rewards in reversed(batch_reward):
            discounted_reward = 0  # The discounted reward so far

            # Iterate starting from last observation. This is a standard way to compute discounted rewards
            for reward in reversed(eps_rewards):
                discounted_reward = reward + (self.params.gamma * discounted_reward)
                batch_reward_to_go.insert(0, discounted_reward)

        batch_reward_to_go = torch.tensor(batch_reward_to_go, dtype=torch.float)

        return batch_reward_to_go

    def rollout(self):
        # collect batch data
        batch_obs = []             # batch observations
        batch_action = []         # batch actions
        batch_log_prob = []       # log probs of each action
        batch_rewards = []         # batch rewards
        batch_rewards_to_go = []   # batch rewards-to-go
        batch_eps_lengths = []     # episode lengths in batch

        t_so_far_in_batch = 0
        while t_so_far_in_batch < self.params.timesteps_per_batch:
            eps_rewards = []
            obs = self.env.reset()
            done = False

            for t_eps in range(self.params.max_timesteps_per_episode):
                # render environment in the first episode
                if (self.params.ppo_render is True) and (len(batch_eps_lengths) == 0):
                    self.env.render()
                # Increment timesteps ran in this batch so far
                t_so_far_in_batch += 1

                # collect observations
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)

                # collect reward, action and log_prob
                eps_rewards.append(reward)
                batch_action.append(action)
                batch_log_prob.append(log_prob)

                if done:
                    break

            batch_eps_lengths.append(t_eps + 1)  # plus 1 because timestep starts at 0
            batch_rewards.append(eps_rewards)

        # convert data to torch tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_action = torch.tensor(batch_action, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
        batch_rewards_to_go = self.compute_rtgs(batch_rewards)

        # Log the episodic returns and episode lengths in this batch
        self.logger.batch_rewards = batch_rewards

        return batch_obs, batch_action, batch_log_prob, batch_rewards_to_go, batch_eps_lengths

    def evaluate(self, batch_obs, batch_action):
        # Query critic network for a value V for each obs in batch_obs
        lif1_spk_rec, lif2_spk_rec, lif3_mem_rec = self.critic(batch_obs)
        # potential at the last time step is interpreted as value
        V = lif3_mem_rec[-1].squeeze()

        # Calculate the log probabilities of batch_actions using most recent actor network
        lif1_spk_rec, lif2_spk_rec, lif3_mem_rec = self.actor(batch_obs)
        # potential at last time step is considered as mean of the Multivariate Normal Distribution
        mean = lif3_mem_rec[-1]
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_action)

        return V, log_prob
    
    def learn(self, total_timesteps):
        print('Training ...')
        t_so_far = 0  # timesteps simulated so far
        # number of times model has been updated (multiple update steps using a single rollout batch in PPO is considered one step)
        self.logger.n_updates = 0

        while t_so_far < total_timesteps:
            # collect a batch of simulations
            batch_obs, batch_action, batch_log_prob, batch_rewards_to_go, batch_eps_length = self.rollout()

            # Add timesteps in current batch to t_so_far
            t_so_far += np.sum(batch_eps_length)

            # calculate value for the current batch of observations
            V, _ = self.evaluate(batch_obs, batch_action)

            # calculate advantage
            A = batch_rewards_to_go - V.detach()

            # normalize advantage
            A = (A - A.mean()) / (A.std() + 1e-10)

            for _ in range(self.params.ppo_n_updates_per_batch):
                # calculate PPO ratio
                V, current_log_prob = self.evaluate(batch_obs, batch_action)
                ratio = torch.exp(current_log_prob - batch_log_prob)

                # calculate the two surrogate losses
                surrogate_loss1 = ratio * A
                surrogate_loss2 = torch.clamp(ratio, 1 - self.params.ppo_clip, 1 + self.params.ppo_clip) * A

                # actor loss
                actor_loss = (- torch.min(surrogate_loss1, surrogate_loss2).mean())

                # critic loss
                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)

                # Calculate gradients and perform backpropagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backpropagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

                # Log actor loss
                self.logger.actor_loss.append(actor_loss.detach().numpy().item())

            # Increment count of updates
            self.logger.n_updates += 1

            # print logged information
            self.log_summary()

            if (self.logger.n_updates % self.params.save_freq) == 0:
                torch.save(self.actor.state_dict(), './snn_ppo_actor.pth')
                torch.save(self.critic.state_dict(), './snn_ppo_critic.pth')

    def log_summary(self):
        # compute average episode rewards and actor loss
        avg_eps_rewards = np.mean([np.sum(eps_rewards) for eps_rewards in self.logger.batch_rewards])
        avg_actor_loss = np.mean(self.logger.actor_loss)

        print(f'-------------------Iteration #{self.logger.n_updates}------------------------')
        print(f'Average reward: {avg_eps_rewards:.2f}')
        print(f'Actor loss: {avg_actor_loss:.2f}')
        print('\n')

        # reset actor loss
        self.logger.actor_loss = []

