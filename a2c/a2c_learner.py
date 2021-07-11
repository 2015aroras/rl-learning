

from typing import Any, List, Tuple
from gym.spaces.space import Space
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn import functional as F
from shared.learner import Learner


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.fc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.pol_output_layer = nn.Linear(hidden_dim, output_dim)
        self.val_output_layer = nn.Linear(hidden_dim, output_dim)

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


class A2CLearner(Learner):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 discount: float = 0.9,
                 lr: float = 0.01):

        super().__init__(observation_space, action_space)

        self.discount = discount
        self.lr = lr

        input_dim: int = self._get_observation_space_dim()
        output_dim: int = self._get_action_space_dim()
        hidden_dim = max(input_dim, output_dim)

        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)

    def get_action(self, state: Any) -> Tuple[Any, Tensor]:
        raise NotImplementedError()

    def update_policy(self, rewards: List[float], action_probs: List[Tensor]):
        raise NotImplementedError()
