from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
from colossalai.tensor import ColoParameter
from colossalai.utils import get_current_device


def process_moe_model(model: nn.Module):
    moe_numel = 0
    no_moe_numel = 0
    for name, module in model.named_modules():
        for p in module.parameters(recurse=False):
            assert isinstance(p, ColoParameter), "parameter `{}` should be initialized corecctly.".format(name)

            if hasattr(p, "moe_info"):
                moe_numel += p.numel()
                p.set_process_group(p.moe_info.pg)
            else:
                no_moe_numel += p.numel()

    buffer = torch.tensor([moe_numel], device=get_current_device())
    dist.all_reduce(buffer)

    moe_numel = int(buffer[0])
    total_numel = moe_numel + no_moe_numel
    return total_numel


class LargeCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(
            weight,
            size_average,
            ignore_index,
            reduce,
            reduction,
            label_smoothing
        )
        assert weight is None
        assert ignore_index == -100
        assert reduction == 'mean'

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:

        assert inputs.ndim == 2
        assert target.ndim == 1
        assert inputs.is_contiguous()
        assert target.is_contiguous()
        assert inputs.size(0) == target.size(0)

        n_batch, n_label = inputs.size()
        max_batch = 2147483648 // n_label

        segment = n_batch
        while segment >= max_batch:
            assert segment % 2 == 0
            segment >>= 1
        n_seg = n_batch // segment
        sum_loss = 0
        for i in range(n_seg):
            begin_idx = i * segment
            seg_inputs = inputs[begin_idx: begin_idx + segment, :]
            seg_target = target[begin_idx: begin_idx + segment]
            seg_loss = F.cross_entropy(seg_inputs, seg_target, weight=self.weight,
                                       ignore_index=self.ignore_index, reduction=self.reduction,
                                       label_smoothing=self.label_smoothing)
            sum_loss += seg_loss

        sum_loss = sum_loss / n_seg
        return sum_loss
