from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces.space import Space
from torch import nn
from torch.functional import Tensor
from torch.nn.parameter import Parameter

from shared.utils import Utils
from shared.learner import Learner


class PolicyNetwork(nn.Module):
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

        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.discount = discount
        self.lr = lr

    def get_action(self, state: Any) -> Tuple[Any, Tensor]:
        nn_input: Tensor = self._parse_state(state).float()
        output_probabilities: Tensor = torch.reshape(
            self.policy_network.forward(nn_input),
            self.discretized_action_space.shape)

        action: List[Any] = []
        action_probability: Tensor = torch.tensor(1.0)
        for i_dim in range(self.discretized_action_space.shape[0]):
            np_out_probs: np.ndarray = output_probabilities[i_dim].detach().numpy()

            index: int = np.random.choice(
                np.arange(np.size(np_out_probs)),
                p=np_out_probs)

            dim_action = self.discretized_action_space[i_dim, index]
            action.append(dim_action.item())

            action_probability *= output_probabilities[i_dim, index]

        # If action space is 0d, change to 0d
        if self.action_space.shape == tuple():
            action = action[0]

        return (action, action_probability)

    def update_policy(self, rewards: List[float], action_probs: List[Tensor]):
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
