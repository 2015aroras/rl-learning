import enum
import logging
import typing
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.core import Env

from learner.a2c_learner import A2CLearner
from learner.reinforce_learner import ReinforceLearner
from learner.shared import Learner
from run.learner_tester import LearnerTester

DEFAULT_ENV_NAME: str = 'CartPole-v0'
DEFAULT_LOG_LEVEL: str = "WARN"
DEFAULT_MAX_EPISODE_LENGTH: int = 500
DEFAULT_EPISODE_COUNT: int = 200
DEFAULT_LR: float = 0.01
DEFAULT_DISCOUNT: float = 0.9

# ENV = 'Acrobot-v1'
# ENV = 'BipedalWalker-v3'
# ENV = 'CartPole-v0'
# ENV = 'Pendulum-v0'
# ENV = 'FrozenLake-v0'
# ENV = 'HalfCheetah-v2'
# ENV = 'MountainCar-v0'
# ENV = 'MountainCarContinuous-v0'


class LearnerType(enum.Enum):
    A2C = enum.auto()
    REINFORCE = enum.auto()


def test_learner(learner_type: LearnerType, args: Namespace) -> None:
    episode_rewards: List[float] = []
    env: Env = gym.make(args.env_name)
    learner: Learner
    if learner_type == LearnerType.A2C:
        learner = A2CLearner(
            env.observation_space,
            env.action_space,
            discount=args.discount,
            lr=args.lr)
    elif learner_type == LearnerType.REINFORCE:
        learner = ReinforceLearner(
            env.observation_space,
            env.action_space,
            discount=args.discount,
            lr=args.lr)
    else:
        raise NotImplementedError()

    learner_tester = LearnerTester(
        learner,
        args.episode_count,
        args.max_episode_length)

    episode_rewards = learner_tester.test(env)
    env.close()

    past_10_rewards_mean = [np.mean(episode_rewards[t-9:t+1])
                            for t in range(len(episode_rewards))]

    plt.plot(episode_rewards)
    plt.plot(past_10_rewards_mean)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def get_arg_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("learner", type=str,
                        help="learner to test")
    parser.add_argument("-e", "--env", dest="env_name", type=str,
                        help="test environment", default=DEFAULT_ENV_NAME)
    parser.add_argument("-l", "--log", dest="log_level", type=str,
                        help="minimum log level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument("-m", "--max-time", dest="max_episode_length",
                        type=int, help="max time in an episode",
                        default=DEFAULT_MAX_EPISODE_LENGTH)
    parser.add_argument("-c", "--count-eps", dest="episode_count", type=int,
                        help="number of episodes",
                        default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("-d", "--discount", dest="discount", type=int,
                        help="discount of rewards",
                        default=DEFAULT_DISCOUNT)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
                        help="learning rate",
                        default=DEFAULT_LR)
    return parser


def main():
    parser: ArgumentParser = get_arg_parser()
    args: Namespace = parser.parse_args()

    log_level: Optional[int] = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(level=log_level)

    learner_type: Optional[LearnerType] = getattr(LearnerType, args.learner.upper(), None)
    if not isinstance(learner_type, LearnerType):
        raise ValueError('Invalid learner type: %s' % args.learner)
    learner_type = typing.cast(LearnerType, learner_type)

    test_learner(learner_type, args)


if __name__ == "__main__":
    main()
