from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces.space import Space
from torch import nn
from torch.functional import Tensor
from torch.nn.parameter import Parameter

from shared.episode_state import EpisodeState
from shared.learner import Learner
from shared.utils import Utils


class _PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.fc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, nn_input: Tensor) -> Tensor:
        hidden_layer_input: Tensor = F.relu(self.fc_layer1.forward(nn_input))
        return F.softmax(self.fc_layer2.forward(hidden_layer_input), -1)

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.fc_layer1.zero_grad(set_to_none=set_to_none)
        self.fc_layer2.zero_grad(set_to_none=set_to_none)


class _ActionData():
    def __init__(self,
                 time: int,
                 probability: Optional[Tensor] = None,
                 reward: Optional[float] = None):
        self.__time = time
        self.__probability = probability
        self.__reward = reward

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


class ReinforceLearner(Learner):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 hidden_dim: Optional[int] = None,
                 discount: float = 0.9,
                 lr: float = 0.01):

        super().__init__(observation_space, action_space)

        input_dim: int = self._get_observation_space_dim()
        output_dim: int = self._get_action_space_size()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)

        self.policy_network = _PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.discount = discount
        self.lr = lr

        self.episode_state: EpisodeState = EpisodeState.FINISHED
        self.episode_actions: List[_ActionData] = []

    def start_episode(self) -> None:
        if self.episode_state is not EpisodeState.FINISHED:
            raise RuntimeError('Cannot start new episode while previous episode is unfinished.')

        self.episode_state = EpisodeState.IN_PROGRESS
        self.episode_actions = []

    def set_time_step_reward(self, time: int, reward: float) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot set reward while episode is not in progress.')

        if len(self.episode_actions) < time:
            raise RuntimeError(f'No action has been taken for the given time: {time}')

        action_data: _ActionData = self.episode_actions[time]
        action_data.set_reward(reward)

    def end_episode(self) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot end episode if episode is not in progress.')

        self.__update_policy()

        self.episode_state = EpisodeState.FINISHED
        self.episode_actions = []

    def get_action(self, state: Any) -> Any:
        nn_input: Tensor = self._parse_state(state).float()
        output_probabilities: Tensor = self.policy_network.forward(nn_input)

        np_out_probs: np.ndarray = output_probabilities.detach().numpy()
        action_index: int = np.random.choice(
            np.arange(np.size(np_out_probs)),
            p=np_out_probs)

        action = self._get_action_from_index(action_index)

        time: int = len(self.episode_actions)
        action_probability: Tensor = output_probabilities[action_index]
        action_data: _ActionData = _ActionData(time, probability=action_probability)
        self.episode_actions.append(action_data)

        return action

    def __update_policy(self):
        rewards: List[float] = [action.get_reward() for action in self.episode_actions]
        action_probs: List[Tensor] = [action.get_probability() for action in self.episode_actions]

        utilities: Tensor = Utils.get_utilities(rewards, self.discount)
        whitened_utilities: Tensor = (utilities - utilities.mean()) / (utilities.std() + 1e-9)

        # Determine the weight changes needed using backprop
        weight_log_grads_by_time: List[Dict[Parameter, Tensor]] = []
        for t in range(len(rewards)):
            action_log_prob: Tensor = torch.log(action_probs[t])
            self.policy_network.zero_grad()
            action_log_prob.backward()

            weight_log_grads_by_time.append({})
            for weight in self.policy_network.parameters():
                if weight in weight_log_grads_by_time[t]:
                    raise RuntimeError(f'Param already processed: {weight}')
                if weight.grad is None:
                    raise RuntimeError(f'Backprop not yet called for {weight}')

                log_grad: Tensor = weight.grad.clone().detach()
                weight_log_grads_by_time[t][weight] = log_grad

        # Update the weights
        for t in range(len(rewards)):
            weight_log_grads = weight_log_grads_by_time[t]
            utility = whitened_utilities[t]
            for weight in self.policy_network.parameters():
                with torch.no_grad():
                    weight += self.lr * utility * weight_log_grads[weight]
