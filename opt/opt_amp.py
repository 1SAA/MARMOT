import psutil
import torch
import torch.nn as nn
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig, OPTForCausalLM
from time import time
from functools import partial
from transformers.modeling_utils import no_init_weights
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
import gc, sys, os

from utils import get_model_size
from colossalai.amp import convert_to_apex_amp
from torch.nn.parallel import DistributedDataParallel as DDP


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len,
                                                vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


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


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_4b(checkpoint=True):
    return GPTLMModel(hidden_size=2304, num_layers=64, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_6b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_12b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=60, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_14b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=70, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_28b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=35, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_32b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_36b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=45, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_8b(checkpoint=True):
    return GPTLMModel(hidden_size=3072, num_layers=72, num_attention_heads=24, checkpoint=checkpoint)


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


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f} MB, CPU memory {:.2f} MB'.format(
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
    BATCH_SIZE = 8
    SEQ_LEN = 2048
    VOCAB_SIZE = 50257
    NUM_STEPS = 6

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    dist.init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')
    torch.cuda.set_device(rank)
    debug_print([0], get_mem_info())

    model = opt_2b(checkpoint=True).cuda()
    numel = get_model_size(model)
    debug_print([0], f'Model numel: {numel}')
    debug_print([0], get_mem_info())
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=2 ** 5)
    model, optimizer = convert_to_apex_amp(model, optimizer, amp_config)
    model = DDP(model, device_ids=[dist.get_rank()])

    model.train()

    result_list = []

    def one_turn():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        start = time()
        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)['logits']
        loss = criterion(outputs, input_ids)
        debug_print([0], get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '))
        fwd_end = time()
        fwd_time = fwd_end - start

        optimizer.backward(loss)
        debug_print([0], get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '))
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        optimizer.step()
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
