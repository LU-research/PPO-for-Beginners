class PPO:
    def __init__(self, policy_class, env, params) -> None:
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(params['beta'], self.obs_dim, params['n_hidden'], self.act_dim)
        self.critic = policy_class(params['beta'], self.obs_dim, params['n_hidden'], 1)