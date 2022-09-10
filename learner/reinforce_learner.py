from typing import Any, List, Optional

import numpy as np
import torch
from gym.spaces.space import Space
from torch import nn
from torch.functional import Tensor
from torch.utils.tensorboard import SummaryWriter

from learner.shared.base_learner import Learner
from learner.shared.episode_state import EpisodeState
from learner.shared.utils import Utils

DEFAULT_DISCOUNT: float = 0.9
DEFAULT_LR: float = 0.01


class _PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, nn_input: Tensor) -> Tensor:
        return self.network.forward(nn_input)


class _ActionData():
    def __init__(self,
                 time: int,
                 prob_logit: Optional[Tensor] = None,
                 reward: Optional[float] = None):
        self.__time = time
        self.__prob_logit = prob_logit
        self.__reward = reward

    def get_time(self) -> int:
        return self.__time

    def get_prob_logit(self) -> Tensor:
        if self.__prob_logit is None:
            raise RuntimeError('prob_logit is not initialized.')

        return self.__prob_logit

    def set_prob_logit(self, prob_logit: Tensor) -> None:
        self.__prob_logit = prob_logit

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
                 tb_writer: SummaryWriter,
                 hidden_dim: Optional[int] = None,
                 discount: Optional[float] = None,
                 lr: Optional[float] = None):

        super().__init__(observation_space, action_space, tb_writer)

        input_dim: int = self._get_observation_space_dim()
        output_dim: int = self._get_action_space_size()
        hidden_dim = max(input_dim, output_dim) if hidden_dim is None else hidden_dim

        self.policy_network = _PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.discount = DEFAULT_DISCOUNT if discount is None else discount
        self.lr = DEFAULT_LR if lr is None else lr
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), self.lr)

        self.episode_state: EpisodeState = EpisodeState.FINISHED
        self.episode_actions: List[_ActionData] = []

    def start_episode(self) -> None:
        if self.episode_state is not EpisodeState.FINISHED:
            raise RuntimeError('Cannot start new episode while previous episode is unfinished.')

        self.episode_state = EpisodeState.IN_PROGRESS
        self.episode_actions = []

    def set_last_action_results(self,
                                reward: float,
                                observation: Any,
                                done: bool) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot set reward while episode is not in progress.')

        action_data: _ActionData = self.episode_actions[-1]
        action_data.set_reward(reward)

    def end_episode(self) -> None:
        if self.episode_state is not EpisodeState.IN_PROGRESS:
            raise RuntimeError('Cannot end episode if episode is not in progress.')

        self.__update_policy()

        self.episode_state = EpisodeState.FINISHED
        self.episode_actions = []

    def get_action(self, state: Any) -> Any:
        nn_input: Tensor = self._parse_state(state).float()
        output_logits: Tensor = self.policy_network.forward(nn_input)

        action_index: int = np.random.choice(
            output_logits.numel(),
            p=torch.softmax(output_logits, -1).detach().numpy())

        action = self._get_action_from_index(action_index)

        time: int = len(self.episode_actions)
        action_logit: Tensor = output_logits[action_index]
        action_data: _ActionData = _ActionData(time, prob_logit=action_logit)
        self.episode_actions.append(action_data)

        return action

    def __update_policy(self):
        rewards: List[float] = [action.get_reward() for action in self.episode_actions]
        action_logits: List[Tensor] = [action.get_prob_logit() for action in self.episode_actions]

        utilities: Tensor = Utils.get_utilities(rewards, self.discount)
        whitened_utilities: Tensor = (utilities - utilities.mean()) / (utilities.std() + 1e-9)

        self.optimizer.zero_grad()

        for t in range(len(rewards)):
            action_log_prob: Tensor = torch.log_softmax(action_logits[t], -1)
            utility = whitened_utilities[t]
            timestep_reward_grad = self.lr * utility * action_log_prob
            timestep_reward_grad.backward()

        self.optimizer.step()
