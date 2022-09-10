import abc
import logging
from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from torch import Tensor
from torch.types import Number
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

DISCRETIZED_DIM_STEPS: int = 5


class Learner(abc.ABC):
    '''
    Base class for a Reinforcement Learner.
    '''

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 tb_writer: SummaryWriter,
                 discretized_dim_steps: int = DISCRETIZED_DIM_STEPS):

        self.observation_space = observation_space
        self.action_space = action_space
        self.tb_writer = tb_writer
        self.discretized_dim_steps = discretized_dim_steps

    @abstractmethod
    def start_episode(self) -> None:
        '''
        Indicates to the learner that a new episode has started.
        '''

    @abstractmethod
    def get_action(self, state: Any) -> Any:
        '''
        Gets the action chosen by the learner for the given state.
        '''

    @abstractmethod
    def set_last_action_results(self,
                                reward: float,
                                observation: Any,
                                done: bool) -> None:
        '''
        Tells the learner the result of the last action given by the learner.
        '''

    @abstractmethod
    def end_episode(self) -> None:
        '''
        Indicates to the learner that the current episode has ended.
        '''

    def _get_observation_space_dim(self) -> int:
        space: Space = self.observation_space
        if isinstance(space, Discrete):
            return 1
        if isinstance(space, Box):
            return space.shape[0]

        raise RuntimeError(f'Unsupported space: {type(space)}')

    def _get_action_space_size(self) -> int:
        space: Space = self.action_space
        if isinstance(space, Discrete):
            return space.n
        if isinstance(space, Box):
            if len(space.shape) != 1:
                raise RuntimeError('Only supporting 1d boxes for now')

            size: int = int(self.discretized_dim_steps ** np.prod(space.shape))
            if size > 100:
                logger.warning('Action space size is %i.', size)

            return size

        raise RuntimeError(f'Unsupported space: {type(space)}')

    def _parse_state(self, state: Any) -> Tensor:
        observation_space: Space = self.observation_space
        if isinstance(observation_space, Discrete):
            return torch.tensor([state])
        if isinstance(observation_space, Box):
            return torch.tensor(state)

        raise RuntimeError(f'Unsupported space: {type(observation_space)}')

    def _get_action_from_index(self, action_index: int) -> Union[Number, List[Number]]:
        space: Space = self.action_space
        if isinstance(self.action_space, Discrete):
            return action_index
        if isinstance(self.action_space, Box):
            if len(space.shape) != 1:
                raise RuntimeError('Only supporting 1d boxes for now')

            dim_steps: int = self.discretized_dim_steps
            n_dim: int = space.shape[0]
            action: List[Number] = []
            for i_dim in range(n_dim):
                dim_index = (action_index // (dim_steps ** (n_dim - 1 - i_dim))) % dim_steps
                dim_action_space: np.ndarray = np.linspace(
                    space.low[i_dim], space.high[i_dim], num=dim_steps, axis=-1)
                action.append(dim_action_space[dim_index])

            return np.atleast_1d(np.squeeze(action)).tolist()

        raise RuntimeError(f'Unsupported space: {type(space)}')
