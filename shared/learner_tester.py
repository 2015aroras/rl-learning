from typing import List

import numpy as np
from gym.core import Env
from torch import Tensor

from shared.learner import Learner


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
            observation: np.ndarray = env.reset()
            rewards: List[float] = []
            action_probs: List[Tensor] = []

            for t in range(self.max_episode_length):
                env.render()
                action, prob = self.learner.get_action(observation)
                # print(observation, action)
                observation, reward, done, _ = env.step(action)
                # print(observation, reward)

                action_probs.append(prob)
                rewards.append(reward)

                if done or t == self.max_episode_length - 1:
                    episode_rewards.append(sum(rewards))
                    print(f"Episode {i_episode} finished after {t + 1} steps")
                    print(f"Episode {i_episode} total reward: {sum(rewards)}")

                    self.learner.update_policy(rewards, action_probs)
                    break

        env.close()
        return episode_rewards
