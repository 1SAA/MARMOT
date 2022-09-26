import psutil
from time import time
from functools import partial

import torch
import torch.nn as nn

import colossalai
from colossalai.context import MOE_CONTEXT
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.loss import MoeLoss
from colossalai.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.gemini import GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.utils import colo_set_process_memory_fraction

from transformers.modeling_utils import no_init_weights
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
import gc, sys, os
from titans.loss.lm_loss import GPTLMLoss
from titans.model.moe import prmoe_4b, prmoe_31b
from utils import process_moe_model


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=get_current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB'.format(
        prefix, get_cur_gpu_mem(), get_gpu_mem(), get_cpu_mem()
    )


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def main():
    BATCH_SIZE = 32
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 5
    PLACEMENT_POLICY = 'const'

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    colo_set_process_memory_fraction(0.4)
    logger = get_dist_logger()

    MOE_CONTEXT.setup(42)
    logger.info(get_mem_info(), ranks=[0])
    # build PR-MOE model
    with ColoInitContext(device=get_current_device()):
        model = prmoe_4b(use_residual=True, checkpoint=True)
    numel = process_moe_model(model)
    logger.info(f'Model numel: {numel}', ranks=[0])
    logger.info(get_mem_info(), ranks=[0])
    # logger.info([p.numel() for p in model.parameters()], ranks=[0])
    # logger.info({n: p.numel() for n, p in model.named_parameters()}, ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    begin = time()
    config_dict = search_chunk_configuration(
        model=model,
        search_range_mb=64,
        search_interval_byte=1024,
        filter_exlarge_params=True
    )
    span = time() - begin
    print("Time is {:.3f} s.".format(span))
    logger.info(config_dict, ranks=[0])

    chunk_manager = ChunkManager(
        config_dict,
        init_device=torch.device('cpu'))
    gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
    # gemini_manager._placement_policy.set_const_memory_boundary(1024 ** 3)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    # logger.info(chunk_manager, ranks=[0])
    logger.info(get_mem_info(), ranks=[0])

    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5, gpu_margin_mem_ratio=0.0)

    # build criterion
    criterion = MoeLoss(aux_weight=0.01, loss_fn=GPTLMLoss)

    # optimizer
    # optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=0.0,
    #                        nvme_offload_dir='/data/user/offload')
    # optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5, gpu_margin_mem_ratio=0.0)
    # logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()

    def one_turn():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])
        fwd_end = time()
        fwd_time = fwd_end - start

        optimizer.backward(loss)
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s',
            ranks=[0])

    for n in range(NUM_STEPS):
        one_turn()

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
