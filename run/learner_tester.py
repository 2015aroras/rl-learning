from typing import List, Optional

import logging
import numpy as np
from gym.core import Env

from learner.shared import Learner

logger = logging.getLogger(__name__)


class LearnerTester():
    def __init__(self,
                 learner: Learner,
                 episode_count: int,
                 max_episode_length: int,
                 render_mode: str = 'human'):
        self.learner = learner
        self.episode_count = episode_count
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode

    def test(self, env: Env) -> List[float]:
        episode_rewards: List[float] = []

        for i_episode in range(self.episode_count):
            episode_reward: float = self.test_episode(env, i_episode)
            episode_rewards.append(episode_reward)

        return episode_rewards

    def test_episode(self, env: Env, i_episode: int) -> float:
        observation: Optional[np.ndarray] = env.reset()
        rewards: List[float] = []

        self.learner.start_episode()

        t: int = 0
        for t in range(self.max_episode_length):
            env.render()
            action = self.learner.get_action(observation)
            logger.info('Get action result: %s', action)
            observation, reward, done, info = env.step(action)
            logger.info('Observation: %s , Reward: %f , Done: %s',
                        observation, reward, done)
            logger.info('Info: %s', info)
            self.learner.set_time_step_reward(t, reward)

            rewards.append(reward)

            if done:
                observation = None
                break

        episode_reward: float = sum(rewards)
        logger.info('Episode %i finished after %i steps', i_episode, t + 1)
        logger.info('Episode %i total reward: %f', i_episode, episode_reward)

        self.learner.end_episode(observation)
        return episode_reward
