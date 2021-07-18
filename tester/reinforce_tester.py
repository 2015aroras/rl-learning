import logging
from typing import List

import gym
from gym.core import Env
from learner.reinforce_learner import ReinforceLearner

from tester.learner_tester import LearnerTester

logger = logging.getLogger(__name__)


class ReinforceTester(LearnerTester):
    def __init__(self,
                 env_name: str,
                 episode_count: int,
                 max_episode_length: int,
                 render_mode: str = 'human'):

        self.env: Env = gym.make(env_name)
        self.learner: ReinforceLearner = ReinforceLearner(
            self.env.observation_space,
            self.env.action_space
        )
        super().__init__(self.learner)
        self.episode_count = episode_count
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode

    def test(self) -> List[float]:
        '''
        Tests the learner.
        '''
        episode_rewards: List[float] = []

        for i_episode in range(self.episode_count):
            episode_reward: float = self._test_episode(self.env,
                                                       self.max_episode_length,
                                                       i_episode,
                                                       self.render_mode)
            episode_rewards.append(episode_reward)

        return episode_rewards
