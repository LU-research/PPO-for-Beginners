class PPO:
    def __init__(self, policy_class, env, params) -> None:
        # Extract environment information
        self.env = env
        self.params = params
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(params.beta, self.obs_dim, params.n_hidden, self.act_dim)
        self.critic = policy_class(params.beta, self.obs_dim, params.n_hidden, 1)

    def get_action(self):
        pass

    def rollout(self):
        # collect batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []         # batch rewards
        batch_rewards_to_go = []   # batch rewards-to-go
        batch_lens = []            # episode lengths in batch

        t_so_far_in_batch = 0
        while t_so_far_in_batch < self.params.timesteps_per_batch:
            obs = self.env.reset()
            done = False

            for eps_t in range(self.params.max_timesteps_per_episode):
                # Increment timesteps ran in this batch so far
                t_so_far_in_batch += 1

                # collect observations
                batch_obs.append(obs)


        obs = self.env.reset()
        done = False

        for ep_t in range(self.params.max_timesteps_per_episode):
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)

            if done:
                break

    def learn(self, total_timesteps):
        t_so_far = 0  # timesteps simulated so far

        while t_so_far < total_timesteps:
            pass