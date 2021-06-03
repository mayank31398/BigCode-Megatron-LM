# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP
from torch.distributed import BarrierOptions, GroupMember

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared
from megatron import get_num_microbatches

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_tflops(batch_size, elapsed_time_per_iteration):
    """Get tflop/s/GPU from global-batch-size and elapsed-time"""
    args = get_args()
    seq_len = args.seq_length
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # Compute throughput.
    samples_per_sec = batch_size / elapsed_time_per_iteration
    tokens_per_sec = samples_per_sec * seq_len

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3, but for 200B model, activation check-pointing will always be on.
    checkpoint_activations_factor = 4 if args.recompute_granularity == 'full' else 3
    coefficient_h_squared = 24
    # GLU activations double the hidden states in the upscaling feed-forward in each transformer layer
    # This leads to 16bsh^2 instead of 8bsh^2 per first feed-forward layer in MLP, thus we increase the coefficient_h_squared by 8.
    # Refer to https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/283#issue-1260805063 for more details.
    if args.glu_activation :
        coefficient_h_squared += 8 

    # In MultiQuery attention, keys and values are shared across heads
    # qkv projection: 6Bsh^2 -> 2Bsh^2 + 4Bshd_kv
    # The formula in https://arxiv.org/pdf/2104.04473.pdf becomes:
    # 4 * (20 Bsh^2 + 4Bshd_kv + 4Bs^2h) = 4*20*Bsh^2 (1 + (d_kv+s)/5h)
    if args.attention_head_type == 'multiquery':
        coefficient_h_squared -= 4 # We substract 4 because of shared kv projection

    # Feed-forward and projections
    flops_per_iteration = (coefficient_h_squared * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2))
    # Attention-matrix computation
    flops_per_iteration += (4 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (seq_len / hidden_size)
    # LM-head
    flops_per_iteration += (6 * batch_size * seq_len * num_layers * (hidden_size**2)) * (vocab_size / (num_layers * hidden_size))

    if args.attention_head_type == 'multiquery':
        d_kv = args.kv_channels
        flops_per_iteration += (4 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (d_kv / hidden_size)  # TODO: maybe tp_size factor missing here

    tflops = flops_per_iteration / (elapsed_time_per_iteration * args.world_size * (10**12))
    return tflops

# TODO check if this is needed
def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions/(1e9)

# TODO check if this is needed
def flops_calculator(model, args, iteration_time):
    gpus_per_model = torch.distributed.get_world_size(group = mpu.get_model_parallel_group())

    approx_parameters_in_billions = get_parameters_in_billions(model)

    batch_size = args.micro_batch_size * get_num_microbatches()

    giga_flops_per_model_per_train_step = approx_parameters_in_billions * batch_size * args.seq_length * 2.0 * 4.0

    effective_tera_flops_per_gpu = giga_flops_per_model_per_train_step / (iteration_time * 1000.0 * gpus_per_model)

    print_rank_0(f"Effective Tera Flops per GPU: {round(effective_tera_flops_per_gpu, 2)} and total parameters {round(approx_parameters_in_billions, 3)} B")
