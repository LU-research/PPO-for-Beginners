import gym


def train(env, params):
    print('Training ...')


if __name__ == '__main__':
    hyper_params = {}

    # Create environment
    pendulum_env = gym.make('Pendulum-v0')

    train(env=pendulum_env, params=hyper_params)