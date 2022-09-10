import logging
from typing import Generator

import gym
from gym.core import Env
from learner.reinforce_learner import ReinforceLearner
from torch.utils.tensorboard import SummaryWriter

from tester.learner_tester import LearnerTester

LOG_PATH = './tb_logs/reinforce/'

logger = logging.getLogger(__name__)


class ReinforceTester(LearnerTester):
    def __init__(self,
                 env_name: str,
                 episode_count: int,
                 max_episode_length: int,
                 render_mode: str = 'human'):
        self.tb_writer: SummaryWriter = SummaryWriter(LOG_PATH, flush_secs=1)
        self.env: Env = gym.make(env_name)
        self.learner: ReinforceLearner = ReinforceLearner(
            self.env.observation_space,
            self.env.action_space,
            self.tb_writer
        )
        super().__init__(self.learner)

        self.episode_count = episode_count
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode

    def test(self) -> Generator[float, None, None]:
        '''
        Tests the learner.
        '''
        for i_episode in range(self.episode_count):
            reward = self._test_episode(self.env,
                                        self.max_episode_length,
                                        i_episode,
                                        self.render_mode)
            yield reward

            self.tb_writer.add_scalar('ep_reward', reward, global_step=i_episode)
