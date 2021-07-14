from typing import Any, Dict, List, Optional, Tuple
import typing

import numpy as np
import torch
from gym.spaces.space import Space
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from shared.episode_state import EpisodeState
from shared.learner import Learner


PolicyUpdate = Dict[Parameter, Tensor]
ValueUpdate = Dict[Parameter, Tensor]


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.fc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.pol_output_layer = nn.Linear(hidden_dim, output_dim)
        self.val_output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, nn_input: Tensor) -> Tensor:
        hidden_layer_output: Tensor = F.relu(self.fc_layer1.forward(nn_input))

        pol_output: Tensor = F.softmax(
            self.pol_output_layer.forward(hidden_layer_output),
            -1)
        val_output: Tensor = self.val_output_layer.forward(hidden_layer_output)
        return torch.cat((pol_output, val_output))

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad(set_to_none=set_to_none)

        self.fc_layer1.zero_grad(set_to_none=set_to_none)
        self.pol_output_layer.zero_grad(set_to_none=set_to_none)
        self.val_output_layer.zero_grad(set_to_none=set_to_none)


class _ActionData():
    def __init__(self,
                 time: int,
                 probability: Optional[Tensor] = None,
                 reward: Optional[float] = None,
                 state_value: Optional[Tensor] = None):
        self.__time = time
        self.__probability = probability
        self.__reward = reward
        self.__state_value = state_value

    def get_time(self) -> int:
        return self.__time

    def get_probability(self) -> Tensor:
        if self.__probability is None:
            raise RuntimeError('Probability is not initialized.')

        return self.__probability

    def set_probability(self, probability: Tensor) -> None:
        self.__probability = probability

    def get_reward(self) -> float:
        if self.__reward is None:
            raise RuntimeError('Reward is not initialized.')

        return self.__reward

    def set_reward(self, reward: float) -> None:
        self.__reward = reward

    def get_state_value(self) -> Tensor:
        if self.__state_value is None:
            raise RuntimeError('State value is not initialized.')

        return self.__state_value

    def set_state_value(self, state_value: Tensor) -> None:
        self.__state_value = state_value.clone().detach().requires_grad_(True)


class _WorkerData(List[_ActionData]):
    def __init__(self):
        super().__init__()
        self.__final_state_value: Optional[Any] = None

    def get_final_state_value(self) -> Any:
        if self.__final_state_value is None:
            raise RuntimeError('Final state value is not initialized.')

        return self.__final_state_value

    def set_final_state_value(self, final_state_value: Tensor) -> None:
        self.__final_state_value = final_state_value


class A2CLearner(Learner):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 max_episode_len: int = 1000,
                 max_workers: int = 5,
                 discount: float = 0.9,
                 lr: float = 0.01):

        super().__init__(observation_space, action_space)

        self.max_episode_len = max_episode_len
        self.discount = discount
        self.lr = lr

        input_dim: int = self._get_observation_space_dim()
        output_dim: int = self._get_action_space_size()
        hidden_dim = max(input_dim, output_dim)

        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)

        self.episode_state: EpisodeState = EpisodeState.FINISHED
        self.worker_data_list: List[_WorkerData] = []
        self.max_workers = max_workers

    def start_episode(self) -> None:
        if self.episode_state is EpisodeState.FINISHED:
            self.episode_state = EpisodeState.IN_PROGRESS
            self.worker_data_list = []
            self.__start_next_worker()
        elif self.episode_state is EpisodeState.IN_PROGRESS:
            self.__start_next_worker()
        else:
            raise RuntimeError(f'Unsupported episode state: {self.episode_state}')

    def __start_next_worker(self) -> None:
        if len(self.worker_data_list) != 0 and len(self.worker_data_list[-1]) == 0:
            raise RuntimeError('Previous worker has done work for no time.')

        worker_data: _WorkerData = _WorkerData()
        self.worker_data_list.append(worker_data)

    def set_time_step_reward(self, time: int, reward: float) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot set reward while episode is not in progress.')

        worker_data: _WorkerData = self.worker_data_list[-1]

        if len(worker_data) < time:
            raise RuntimeError(f'Current worker has not worked for the given time: {time}')

        worker_data[time].set_reward(reward)

    def end_episode(self, final_state: Optional[Any] = None) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot end episode if episode is not in progress.')

        final_state_value: Tensor = torch.tensor(0)
        if final_state is not None:
            _, val_output = self.__forward(final_state)
            final_state_value = val_output
        self.worker_data_list[-1].set_final_state_value(final_state_value)

        if len(self.worker_data_list) >= self.max_workers:
            self.episode_state = EpisodeState.FINISHED
            self.__update_policy()

    def get_action(self, state: Any) -> Any:
        pol_output, val_output = self.__forward(state)
        np_out_probs: np.ndarray = pol_output.detach().numpy()
        action_index: int = np.random.choice(
            np.arange(np.size(np_out_probs)),
            p=np_out_probs)

        worker_data: _WorkerData = self.worker_data_list[-1]
        worker_time: int = len(worker_data)

        action = self._get_action_from_index(action_index)
        action_probability: Tensor = pol_output[action_index]
        action_data: _ActionData = _ActionData(
            worker_time,
            probability=action_probability,
            state_value=val_output)
        worker_data.append(action_data)

        return action

    def __forward(self, state: Any) -> Tuple[Tensor, Tensor]:
        nn_input: Tensor = self._parse_state(state).float()
        nn_output: Tensor = self.policy_network.forward(nn_input)

        split_output = torch.split(nn_output, self._get_action_space_size())
        # print(split_output)
        if len(split_output) != 2:
            raise RuntimeError(f'Split output has incorrect size: {len(split_output)}')
        return typing.cast(Tuple[Tensor, Tensor], split_output)

    def __update_policy(self) -> None:
        if self.episode_state is not EpisodeState.FINISHED:
            raise RuntimeError('Cannot update policy unless episode is finished.')

        worker_updates: List[Tuple[PolicyUpdate, ValueUpdate]] = [
            self.__get_updates_from_worker(worker_data) for worker_data in self.worker_data_list
        ]

        for weight in self.policy_network.parameters():
            with torch.no_grad():
                # print('Weight before:', weight)
                for update in worker_updates:
                    policy_update, value_update = update

                    if weight in policy_update:
                        weight += self.lr * policy_update[weight]

                    if weight in value_update:
                        weight -= self.lr * value_update[weight]

                # print('Weight after:', weight)

        # for weight in self.policy_network.parameters():
        #     print(weight)

    def __get_updates_from_worker(self, worker_data: _WorkerData) -> Tuple[PolicyUpdate, ValueUpdate]:
        policy_update: PolicyUpdate = {}
        value_update: ValueUpdate = {}

        utility: Tensor = worker_data.get_final_state_value()
        for action_data in reversed(worker_data):
            utility = action_data.get_reward() + self.discount * utility
            action_advantage: Tensor = utility - action_data.get_state_value()
            # print('Utility', utility)
            # print('State value', action_data.get_state_value())
            # print('Action advantage:', action_advantage)

            # Policy update
            log_action_prob = torch.log(action_data.get_probability())
            self.policy_network.zero_grad()
            log_action_prob.backward(retain_graph=True)
            for weight in self.policy_network.parameters():
                if weight.grad is None:
                    raise RuntimeError(f'Gradient is none for weight: {weight}')

                weight_update: Tensor = weight.grad * action_advantage
                if weight not in policy_update:
                    policy_update[weight] = weight_update
                else:
                    policy_update[weight] += weight_update

            # Value update
            squared_action_advantage: Tensor = action_advantage ** 2
            self.policy_network.zero_grad()
            squared_action_advantage.backward(retain_graph=True)
            for weight in self.policy_network.parameters():
                if weight.grad is None:
                    raise RuntimeError(f'Gradient is none for weight: {weight}')

                if weight not in value_update:
                    value_update[weight] = weight.grad
                else:
                    value_update[weight] += weight.grad

        return (policy_update, value_update)
