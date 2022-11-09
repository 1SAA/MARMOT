import psutil
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig, OPTForCausalLM
from deepspeed.ops.adam import DeepSpeedCPUAdam
from time import time
from functools import partial
from transformers.modeling_utils import no_init_weights
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
import gc, sys, os
from utils import opt_2b, opt_6b, opt_13b


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB'.format(
        prefix, get_cur_gpu_mem(), get_gpu_mem(), get_cpu_mem()
    )


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def memory_cap(size_in_GB):
    print(f"use only {size_in_GB} GB of CUDA memory")
    assert dist.is_initialized(), "memory_cap must be used after dist init"
    local_rank = dist.get_rank()
    cuda_capacity = torch.cuda.get_device_properties(local_rank).total_memory
    size_in_B = (size_in_GB * 1024 ** 3)
    if size_in_B > cuda_capacity:
        print(f'memory_cap is uselsess since {cuda_capacity / 1024 ** 3} less than {size_in_GB}')
        return
    fraction = (size_in_GB * 1024 ** 3) / cuda_capacity
    print(f'mem faction is {fraction}')
    torch.cuda.set_per_process_memory_fraction(fraction, local_rank)


def debug_print(ranks, *args):
    if dist.get_rank() in ranks:
        print(*args)
    dist.barrier()


def main():
    BATCH_SIZE = 24
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 6

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./zero_config.json')
    total_bs = BATCH_SIZE * int(os.environ['WORLD_SIZE'])
    deepspeed_plugin.deepspeed_config['train_batch_size'] = total_bs
    deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = BATCH_SIZE
    deepspeed_config = deepspeed_plugin.deepspeed_config
    deepspeed.init_distributed()
    memory_cap(40)
    debug_print([0], get_mem_info())

    # build GPT model
    with deepspeed.zero.Init(config_dict_or_path=deepspeed_config):
        model = opt_13b(checkpoint=True)
    numel = deepspeed.runtime.zero.partition_parameters.param_count
    debug_print([0], f'Model numel: {numel}')
    debug_print([0], get_mem_info())

    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=deepspeed_config)
    model.train()

    result_list = []

    def one_turn():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        start = time()
        outputs = model(input_ids, attn_mask)['logits']
        loss = criterion(outputs, input_ids)
        debug_print([0], get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '))
        fwd_end = time()
        fwd_time = fwd_end - start

        model.backward(loss)
        debug_print([0], get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '))
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        model.step()
        debug_print([0], get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '))
        optim_time = time() - bwd_end
        step_time = time() - start
        debug_print(
            [0],
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, '
            f'TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, '
            f'OPTIM time: {optim_time:.3f}s'
        )
        result_list.append(get_tflops_func(step_time))

    for n in range(NUM_STEPS):
        one_turn()

    result_list = result_list[1:]
    result_list.sort()
    assert NUM_STEPS % 2 == 0
    mid_pos = NUM_STEPS // 2 - 1
    debug_print([0], "Profiling ended. The mode TFLOPS: {:.3f}".format(result_list[mid_pos]))

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              schedule=schedule(wait=1, warmup=2, active=2),
    #              on_trace_ready=tensorboard_trace_handler(
    #                  f'opt-6.7b/v3-full-{PLACEMENT_POLICY}-{dist.get_world_size()}gpu'),
    #              record_shapes=True,
    #              profile_memory=True) as prof:
    #     for n in range(NUM_STEPS):
    #         one_turn()
    #         prof.step()
    # dist.barrier()


if __name__ == '__main__':
    main()
