import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig, OPTForCausalLM


def _create_opt(config, checkpoint=True):
    model = OPTForCausalLM(config)
    if checkpoint:
        model.gradient_checkpointing_enable()
    return model


def opt_30b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-30b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_13b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-13b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_6b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-6.7b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_2b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-2.7b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_1b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    return _create_opt(config, checkpoint=checkpoint)


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel
