import abc
import logging
from abc import abstractmethod
from typing import Any, Generator, List, Tuple

import numpy as np
from gym.core import Env

from learner.shared import Learner

logger = logging.getLogger(__name__)


class LearnerTester(abc.ABC):
    def __init__(self, learner: Learner):
        self.learner = learner

    @abstractmethod
    def test(self) -> Generator[float, None, None]:
        '''
        Tests the learner.
        '''

    def _test_episode(self,
                      env: Env,
                      max_episode_length: int,
                      i_episode: int,
                      render_mode: str) -> float:
        observation: np.ndarray = env.reset()
        rewards: List[float] = []

        self.learner.start_episode()

        i_step: int = 0
        for i_step in range(max_episode_length):
            env.render(render_mode)
            observation, reward, done = self._test_step(env, observation)

            rewards.append(reward)

            if done:
                break

        episode_reward: float = sum(rewards)
        logger.info('Episode %i finished after %i steps', i_episode, i_step + 1)
        logger.info('Episode %i total reward: %f', i_episode, episode_reward)

        self.learner.end_episode()
        return episode_reward

    def _test_step(self,
                   env: Env,
                   observation: Any) -> Tuple[Any, float, bool]:
        action = self.learner.get_action(observation)
        logger.info('Get action result: %s', action)

        observation, reward, done, info = env.step(action)
        logger.info('Observation: %s , Reward: %f , Done: %s',
                    observation, reward, done)
        logger.info('Info: %s', info)

        self.learner.set_last_action_results(reward, observation, done)

        return observation, reward, done
