import logging
import typing
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym.spaces.space import Space
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

from learner.shared.base_learner import Learner
from learner.shared.episode_state import EpisodeState

PolicyUpdate = Dict[Parameter, Tensor]
ValueUpdate = Dict[Parameter, Tensor]

logger = logging.getLogger(__name__)


DEFAULT_WORKER_COUNT: int = 5
DEFAULT_DISCOUNT: float = 0.99
DEFAULT_LR: float = 1e-3
DEFAULT_ENTROPY_REGULARIZER: float = 1e-2
DEFAULT_UPDATE_INTERVAL: int = 5


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
                 entropy: Optional[Tensor] = None,
                 reward: Optional[float] = None,
                 state_value: Optional[Tensor] = None):
        self.__time = time
        self.__probability = probability
        self.__entropy = entropy
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

    def get_entropy(self) -> Tensor:
        if self.__entropy is None:
            raise RuntimeError('Entropy is not initialized.')

        return self.__entropy

    def set_entropy(self, entropy: Tensor) -> None:
        self.__entropy = entropy

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
        self.__final_state_value: Optional[Tensor] = None
        self.__episode_state: EpisodeState = EpisodeState.IN_PROGRESS

    def get_episode_state(self) -> EpisodeState:
        return self.__episode_state

    def set_episode_state(self, episode_state: EpisodeState) -> None:
        self.__episode_state = episode_state

    def get_final_state_value(self) -> Tensor:
        if self.__final_state_value is None:
            raise RuntimeError('Final state value is not initialized.')

        return self.__final_state_value

    def set_final_state_value(self, final_state_value: Tensor) -> None:
        self.__final_state_value = final_state_value


class A2CLearner(Learner):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 tb_writer: SummaryWriter,
                 worker_count: Optional[int] = None,
                 discount: Optional[float] = None,
                 lr: Optional[float] = None,
                 entropy_regularizer: Optional[float] = None,
                 update_interval: Optional[int] = None):

        super().__init__(observation_space, action_space, tb_writer)

        self.discount: float = DEFAULT_DISCOUNT if discount is None else discount
        self.lr: float = DEFAULT_LR if lr is None else lr
        self.entropy_regularizer: float = (DEFAULT_ENTROPY_REGULARIZER
                                           if entropy_regularizer is None
                                           else entropy_regularizer)
        self.update_interval: int = (DEFAULT_UPDATE_INTERVAL
                                     if update_interval is None
                                     else update_interval)

        input_dim: int = self._get_observation_space_dim()
        output_dim: int = self._get_action_space_size()
        hidden_dim = max(input_dim, output_dim)

        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)

        self.worker_count: int = DEFAULT_WORKER_COUNT if worker_count is None else worker_count
        self.worker_data_list: List[_WorkerData] = []
        self.__reset_workers()
        self.i_worker: int = 0

    def start_episode(self) -> None:
        # worker: _WorkerData = self.__get_current_worker()
        # if worker.get_episode_state() is EpisodeState.FINISHED:
        #     self.worker_data_list[self.i_worker] = _WorkerData()
        pass

    def set_last_action_results(self,
                                reward: float,
                                observation: Any,
                                done: bool) -> None:
        worker_data: _WorkerData = self.__get_current_worker()
        if worker_data.get_episode_state() is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot set reward while episode is not in progress.')

        worker_data[-1].set_reward(reward)
        if len(worker_data) == self.update_interval and not done:
            _, val_output = self.__forward(observation)
            worker_data.set_final_state_value(val_output)

            self.__next_worker()

    def end_episode(self) -> None:
        worker: _WorkerData = self.__get_current_worker()
        if worker.get_episode_state() is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot end episode if episode is not in progress.')

        worker.set_final_state_value(torch.tensor(0))
        worker.set_episode_state(EpisodeState.FINISHED)
        self.__next_worker()

    def get_action(self, state: Any) -> Any:
        pol_output, val_output = self.__forward(state)
        np_out_probs: np.ndarray = pol_output.detach().numpy()
        action_index: int = np.random.choice(
            np.arange(np.size(np_out_probs)),
            p=np_out_probs)

        worker_data: _WorkerData = self.__get_current_worker()
        worker_time: int = len(worker_data)

        action = self._get_action_from_index(action_index)
        action_probability: Tensor = pol_output[action_index]
        action_data: _ActionData = _ActionData(
            worker_time,
            probability=action_probability,
            entropy=A2CLearner.__get_entropy(pol_output),
            state_value=val_output)
        worker_data.append(action_data)

        return action

    def __get_current_worker(self) -> _WorkerData:
        return self.worker_data_list[self.i_worker]

    def __next_worker(self) -> None:
        if len(self.worker_data_list[self.i_worker]) == 0:
            raise RuntimeError('Previous worker has done work for no time:', self.i_worker)

        if self.i_worker == self.worker_count - 1:
            self.__update_policy()
            self.__reset_workers()

        self.i_worker = (self.i_worker + 1) % self.worker_count

    def __reset_workers(self) -> None:
        self.worker_data_list = [
            _WorkerData() for _ in range(self.worker_count)
        ]

    def __forward(self, state: Any) -> Tuple[Tensor, Tensor]:
        nn_input: Tensor = self._parse_state(state).float()
        nn_output: Tensor = self.policy_network.forward(nn_input)

        split_output = torch.split(nn_output, self._get_action_space_size())
        if len(split_output) != 2:
            raise RuntimeError(f'Split output has incorrect size: {split_output}')
        return typing.cast(Tuple[Tensor, Tensor], split_output)

    def __update_policy(self) -> None:
        if self.i_worker != self.worker_count - 1:
            raise RuntimeError('Cannot update policy until last worker is done. '
                               f'Current worker is {self.i_worker}')

        worker_updates: List[Tuple[PolicyUpdate, ValueUpdate]] = [
            self.__get_updates_from_worker(worker_data) for worker_data in self.worker_data_list
        ]

        for weights in self.policy_network.parameters():
            with torch.no_grad():
                # print('Weight before:', weight)
                for update in worker_updates:
                    policy_update, value_update = update

                    if weights in policy_update:
                        weights += policy_update[weights]

                    if weights in value_update:
                        weights += value_update[weights]

                # print('Weight after:', weight)

        # for weight in self.policy_network.parameters():
        #     print(weight)

    def __get_updates_from_worker(self,
                                  worker_data: _WorkerData
                                  ) -> Tuple[PolicyUpdate, ValueUpdate]:
        policy_update: PolicyUpdate = {}
        value_update: ValueUpdate = {}
        for weights in self.policy_network.parameters():
            policy_update[weights] = torch.zeros(weights.shape)
            value_update[weights] = torch.zeros(weights.shape)

        utility: Tensor = worker_data.get_final_state_value()
        for action_data in reversed(worker_data):
            utility = action_data.get_reward() + self.discount * utility
            action_advantage: Tensor = utility - action_data.get_state_value()
            # print('Action advantage', action_advantage)
            logger.info('Action advantage: %f', action_advantage)
            logger.debug('Policy entropy: %f', action_data.get_entropy())

            # Policy update (without entropy adjustment)
            log_action_prob = torch.log(action_data.get_probability())
            self.policy_network.zero_grad()
            log_action_prob.backward(retain_graph=True)
            for weights in self.policy_network.parameters():
                if weights.grad is None:
                    raise RuntimeError(f'Gradient is none for weights: {weights}')

                weights_update: Tensor = self.lr * weights.grad * action_advantage
                policy_update[weights] += weights_update

            # Entropy adjustment of policy update
            policy_entropy = action_data.get_entropy()
            self.policy_network.zero_grad()
            policy_entropy.backward(retain_graph=True)
            for weights in self.policy_network.parameters():
                if weights.grad is None:
                    raise RuntimeError(f'Gradient is none for weights: {weights}')

                weights_update: Tensor = self.lr * self.entropy_regularizer * weights.grad
                policy_update[weights] += weights_update

            # Value update
            squared_action_advantage: Tensor = action_advantage ** 2
            self.policy_network.zero_grad()
            squared_action_advantage.backward(retain_graph=True)
            for weights in self.policy_network.parameters():
                if weights.grad is None:
                    raise RuntimeError(f'Gradient is none for weight: {weights}')

                weights_update: Tensor = -1 * self.lr * weights.grad
                value_update[weights] += weights_update

        return (policy_update, value_update)

    @staticmethod
    def __get_entropy(pol_output: Tensor) -> Tensor:
        non_zero_pol_output = pol_output[pol_output.nonzero(as_tuple=True)]
        entropy: Tensor = -torch.sum(
            torch.mul(non_zero_pol_output, torch.log(non_zero_pol_output)))
        logger.info('Entropy: %f', entropy)
        return entropy
