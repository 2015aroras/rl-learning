import abc
from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from torch import Tensor

DISCRETIZED_DIM_STEPS: int = 5


class Learner(abc.ABC):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 discretized_dim_steps: int = DISCRETIZED_DIM_STEPS):

        self.observation_space = observation_space
        self.action_space = action_space
        self.discretized_action_space = self._get_discretized_space(
            self.action_space, discretized_dim_steps)

    @abstractmethod
    def get_action(self, state: Any) -> Tuple[Any, Tensor]:
        pass

    @abstractmethod
    def update_policy(self, rewards: List[float], action_probs: List[Tensor]):
        pass

    def _get_observation_space_dim(self) -> int:
        space: Space = self.observation_space
        if isinstance(space, Discrete):
            return 1
        if isinstance(space, Box):
            return space.shape[0]

        raise RuntimeError(f'Unsupported space: {type(space)}')

    def _get_action_space_size(self) -> int:
        return self.discretized_action_space.size

    def _parse_state(self, state: Any) -> Tensor:
        observation_space: Space = self.observation_space
        if isinstance(observation_space, Discrete):
            return torch.tensor([state])
        if isinstance(observation_space, Box):
            return torch.tensor(state)

        raise RuntimeError(f'Unsupported space: {type(observation_space)}')

    @staticmethod
    def _get_discretized_space(space: Space, dim_steps: int) -> np.ndarray:
        if isinstance(space, Discrete):
            return np.array([np.arange(space.n)])
        if isinstance(space, Box):
            if len(space.shape) != 1:
                raise RuntimeError('Only supporting 1d boxes for now')

            return np.array([
                np.linspace(space.low[i], space.high[i], dim_steps, -1)
                for i in range(space.shape[0])])

        raise RuntimeError(f'Unsupported space: {type(space)}')
