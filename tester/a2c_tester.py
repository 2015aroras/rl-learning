from typing import Any, Generator, List

import logging
import typing
import gym
from gym.core import Env

from learner.a2c_learner import A2CLearner
from tester.learner_tester import LearnerTester

logger = logging.getLogger(__name__)


class A2CTester(LearnerTester):
    def __init__(self,
                 env_name: str,
                 episode_count: int,
                 max_episode_length: int,
                 render_mode: str = 'human'):
        self.env_name: str = env_name
        self.main_env: Env = gym.make(env_name)
        learner: A2CLearner = A2CLearner(
            self.main_env.observation_space,
            self.main_env.action_space
        )
        super().__init__(learner)

        self.worker_count: int = learner.worker_count
        self.episode_count: int = episode_count
        self.max_episode_length: int = max_episode_length
        self.render_mode: str = render_mode

    def test(self) -> Generator[float, None, None]:
        workers_env: List[Env] = [self.main_env] + [
            gym.make(self.env_name) for _ in range(self.worker_count - 1)]
        workers_observation: List[Any] = [env.reset() for env in workers_env]
        main_env_curr_episode_reward: List[float] = []

        i_episode: int = 0
        workers_i_time: List[int] = [0 for _ in range(self.worker_count)]
        while i_episode < self.episode_count:
            # Learner updates the worker during learning.
            i_worker: int = typing.cast(A2CLearner, self.learner).i_worker
            worker_env: Env = workers_env[i_worker]
            worker_observation: Any = workers_observation[i_worker]
            worker_i_time: int = workers_i_time[i_worker]

            if worker_i_time == 0:
                self.learner.start_episode()
                worker_observation = worker_env.reset()

            if i_worker == 0:
                worker_env.render(self.render_mode)

            observation, reward, done = self._test_step(
                worker_env, worker_observation)

            workers_observation[i_worker] = observation

            if i_worker == 0:
                main_env_curr_episode_reward.append(reward)

            if done or worker_i_time >= self.max_episode_length - 1:
                self.learner.end_episode()
                i_episode += 1
                workers_i_time[i_worker] = 0
                if i_worker == 0:
                    yield sum(main_env_curr_episode_reward)
                    main_env_curr_episode_reward = []

            else:
                workers_i_time[i_worker] += 1

        for worker_env in workers_env:
            worker_env.close()
