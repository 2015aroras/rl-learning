import enum
import logging
import typing
from argparse import ArgumentParser, Namespace
from typing import Optional

from tester.a2c_tester import A2CTester
from tester.learner_tester import LearnerTester
from tester.reinforce_tester import ReinforceTester

DEFAULT_ENV_NAME: str = 'CartPole-v0'
DEFAULT_LOG_LEVEL: str = "WARN"
DEFAULT_MAX_EPISODE_LENGTH: int = 500
DEFAULT_EPISODE_COUNT: int = 200

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
    learner_tester: LearnerTester
    if learner_type == LearnerType.A2C:
        learner_tester = A2CTester(
            args.env_name,
            args.episode_count,
            args.max_episode_length)
    elif learner_type == LearnerType.REINFORCE:
        learner_tester = ReinforceTester(
            args.env_name,
            args.episode_count,
            args.max_episode_length)
    else:
        raise NotImplementedError(f'Learner type not yet supported: {learner_type}')

    list(learner_tester.test())


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
                        help="discount of rewards")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
                        help="learning rate")
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
