import torch
import torch.nn as nn
import torch.distributed as dist
from colossalai.tensor import ColoParameter
from colossalai.utils import get_current_device


def process_moe_model(model: nn.Module):
    moe_numel = 0
    no_moe_numel = 0
    for name, p in model.named_parameters():
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
