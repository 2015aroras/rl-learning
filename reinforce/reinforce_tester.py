import gym
import matplotlib.pyplot as plt
import numpy as np

from shared.learner_tester import LearnerTester
from reinforce.reinforce_learner import ReinforceLearner

LR = 0.01
DISCOUNT = 0.9
EPISODE_COUNT = 150
MAX_EPISODE_LENGTH = 500
# ENV = 'Acrobot-v1'
ENV = 'BipedalWalker-v3'
# ENV = 'CartPole-v0'
# ENV = 'Pendulum-v0'
# ENV = 'FrozenLake-v0'
# ENV = 'HalfCheetah-v2'
# ENV = 'MountainCar-v0'
# ENV = 'MountainCarContinuous-v0'


def main() -> None:
    env = gym.make(ENV)
    reinforce_learner = ReinforceLearner(
        env.observation_space,
        env.action_space,
        discount=DISCOUNT,
        lr=LR)
    reinforce_tester = LearnerTester(
        reinforce_learner,
        EPISODE_COUNT,
        MAX_EPISODE_LENGTH)
    episode_rewards = reinforce_tester.test(env)
    env.close()

    past_10_rewards_mean = [np.mean(episode_rewards[t-9:t+1])
                            for t in range(len(episode_rewards))]

    plt.plot(episode_rewards)
    plt.plot(past_10_rewards_mean)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == "__main__":
    main()
