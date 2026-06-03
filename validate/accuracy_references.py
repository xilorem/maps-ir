from __future__ import annotations

from typing import Callable, Dict, List

import torch


ReferenceFn = Callable[[List[torch.Tensor]], List[torch.Tensor]]


def matmul_exp_add_reference(inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")
    lhs, rhs, bias = inputs
    return [torch.exp(torch.matmul(lhs, rhs)) + bias]


REFERENCE_REGISTRY: Dict[str, ReferenceFn] = {
    "matmul_exp_add_reference": matmul_exp_add_reference,
}
