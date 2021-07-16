from typing import List

import numpy as np
import torch
from torch.functional import Tensor


class Utils():
    @staticmethod
    def get_utilities(rewards: List[float], discount: float) -> Tensor:
        T = len(rewards)
        discounts: np.ndarray = np.logspace(0, T - 1, T, base=discount)

        utilities: List[Tensor] = []
        for t in range(T):
            utility = np.dot(rewards[t:], discounts[:T - t])
            utilities.append(torch.tensor(utility))

        return torch.stack(utilities)
