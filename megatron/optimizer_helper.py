import torch
from torch import inf
from megatron.optimizer.distrib_optimizer import DistributedOptimizer
import logging
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import os

logging.debug("Logging is configured correctly")

def optimizer_helper_step(optimizer, args, timers):

    # Copy gradients from model params to main params.
    timers('optimizer-copy-to-main-grad', log_level=1).start(barrier=args.barrier_with_L1_time)
    optimizer._copy_model_grads_to_main_grads()
    timers('optimizer-copy-to-main-grad').stop()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if optimizer.grad_scaler:
        assert False, "grad scaler for bucket-wise optimizer step not support"

    # Clip the main gradients.
    timers('optimizer-clip-main-grad', log_level=1).start(barrier=args.barrier_with_L1_time)
    grad_norm = None
    if optimizer.clip_grad > 0.0:
        # ------------------------------ JINDA_DEBUG 1 Clip Grad bucket-wise ------------------------------
        # Option 1: Clip grad bucket wise.
        
        params = optimizer.get_parameters()
        grads_for_norm = optimizer.get_main_grads_for_grad_norm()
        pre_given_total_norm = calculate_pre_given_total_norm(
            params,
            grads_for_norm,
            optimizer.clip_grad,
            optimizer.check_for_nan_in_grad,
            model_parallel_group=optimizer.get_model_parallel_group(),
        )
        grad_norm = pre_given_total_norm
        
        for gbuf_index, grad_buffer in enumerate(optimizer.grad_buffers):
            dtype = grad_buffer.dtype
            for bucket_index, _ in enumerate(grad_buffer.buckets):
                for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
                    bucket_shard_fp32_params_this_group = optimizer.bucket_wise_shard_fp32_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
                    bucket_shard_fp32_from_float16_params_this_group = optimizer.bucket_wise_shard_fp32_from_float16_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
                    param_group['params'] = [
                        *bucket_shard_fp32_params_this_group,
                        *bucket_shard_fp32_from_float16_params_this_group,
                    ]
                params = optimizer.get_parameters()
                grads_for_norm = optimizer.get_main_grads_for_grad_norm()
                clip_grad_norm_fp32_with_pregiven_totalnorm(
                    params,
                    grads_for_norm,
                    optimizer.clip_grad,
                    optimizer.check_for_nan_in_grad,
                    model_parallel_group=optimizer.get_model_parallel_group(),
                    total_norm=pre_given_total_norm,
                )
        
        """Here we need to set param_group back, I don't know why, must be somewhere used. TODO find where used it."""
        for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
            shard_fp32_params_this_group = optimizer.shard_fp32_groups[group_index]
            shard_fp32_from_float16_params_this_group = optimizer.shard_fp32_from_float16_groups[group_index]
            param_group['params'] = [
                *shard_fp32_params_this_group, 
                *shard_fp32_from_float16_params_this_group
            ]

        # Option 2: Clip grad all in once.
        # grad_norm = optimizer.clip_grad_norm(optimizer.clip_grad, optimizer.check_for_nan_in_grad)
        
        # ------------------------------ JINDA_DEBUG 1 Clip Grad bucket-wise ------------------------------

    timers('optimizer-clip-main-grad').stop()

    # Count the zeros in the grads.
    timers('optimizer-count-zeros', log_level=1).start(barrier=args.barrier_with_L1_time)
    num_zeros_in_grad = optimizer.count_zeros() if optimizer.log_num_zeros_in_grad else None
    timers('optimizer-count-zeros').stop()

    # Step the optimizer.
    timers('optimizer-inner-step', log_level=1).start(barrier=args.barrier_with_L1_time)
    # optimizer.optimizer.step()

    # ------------------------- JINDA_DEBUG 2 for bucket-wise optimizer -------------------------

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # Option 1:Step the optimizer for bucket-wise
    if rank == 0:
        print(f'JINDA_DEBUG: Step the optimizer for bucket-wise')
    for gbuf_index, grad_buffer in enumerate(optimizer.grad_buffers):
        dtype = grad_buffer.dtype
        for bucket_index, _ in enumerate(grad_buffer.buckets):
            # optimizer_helper_bucket_wise_inner_step(optimizer, gbuf_index, bucket_index)
            for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
                bucket_shard_fp32_params_this_group = optimizer.bucket_wise_shard_fp32_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
                bucket_shard_fp32_from_float16_params_this_group = optimizer.bucket_wise_shard_fp32_from_float16_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
                param_group['params'] = [
                    *bucket_shard_fp32_params_this_group,
                    *bucket_shard_fp32_from_float16_params_this_group,
                ]
            optimizer.optimizer.step()

            """
            Here to reset step for optimizer, since optimizer.step() innerly increase step by 1, but we only need to increase step by 1 for all bucket once.
            TODO, we can update optimizer step, so optimizer.step() won't update param['step'] innerly.
            """
            for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
                param_group['step'] -= 1
    """Here we need to increase step by 1 after all bucket step finished."""
    for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
        param_group['step'] += 1
    """Here we need to set param_group back, I don't know why, must be somewhere used. TODO find where used it."""
    for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
        shard_fp32_params_this_group = optimizer.shard_fp32_groups[group_index]
        shard_fp32_from_float16_params_this_group = optimizer.shard_fp32_from_float16_groups[group_index]
        param_group['params'] = [
            *shard_fp32_params_this_group, 
            *shard_fp32_from_float16_params_this_group
        ]

    # Option 2: Step the optimizer for all
    # if rank == 0:
    #     print(f'JINDA_DEBUG: Step the optimizer for all')
    # for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
    #     all_params_this_group = []
    #     for gbuf_idx, grad_buffer in enumerate(optimizer.grad_buffers):
    #         dtype = grad_buffer.dtype
    #         for bucket_idx, _ in enumerate(grad_buffer.buckets):
    #             bucket_shard_fp32_params_this_group = optimizer.bucket_wise_shard_fp32_params_this_group.get((group_index, gbuf_idx, bucket_idx), [])
    #             bucket_shard_fp32_from_float16_params_this_group = optimizer.bucket_wise_shard_fp32_from_float16_params_this_group.get((group_index, gbuf_idx, bucket_idx), [])
    #             all_params_this_group = [*all_params_this_group, *bucket_shard_fp32_params_this_group, *bucket_shard_fp32_from_float16_params_this_group]
    #     param_group['params'] = [
    #         *all_params_this_group
    #     ]
    # optimizer.optimizer.step()
    # for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
    #     param_group['params'] = [
    #     ]

    # JINDA_DEBUG print
    # if rank == 0:
    #     print(f'JINDA_DEBUG: optimizer.capturable: {optimizer.optimizer.capturable}, optimizer._dummy_overflow_buf: {optimizer.optimizer._dummy_overflow_buf}')
    #     for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
    #         if 'step' in param_group:
    #             x = param_group['step']
    #         else:
    #             x = 'None'
    #         print(f"JINDA_DEBUG: param_group[{group_index}]['step']: {x}, len(param_group[{group_index}]['params']): {len(param_group['params'])}, param_group[{group_index}]['bias_correction']: {param_group['bias_correction']}")
    
    # ------------------------- JINDA_DEBUG 2 for bucket-wise optimizer -------------------------

    timers('optimizer-inner-step').stop()

    # Update params from main params.
    timers('optimizer-copy-main-to-model-params', log_level=1).start(
        barrier=args.barrier_with_L1_time
    )
    optimizer._copy_main_params_to_model_params()
    timers('optimizer-copy-main-to-model-params').stop()

    optimizer.update_successful = True

    # If not overlapping all-gather for parameters, launch synchronous all-gather
    # communication calls here. If overlapping all-gather for parameters, the following
    # call to _gather_all_model_params is a no-op: the first all-gather is launched
    # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
    # are launched in the forward pre-hook.
    timers('params-all-gather', log_level=1).start(barrier=args.barrier_with_L1_time)
    optimizer._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
    timers('params-all-gather').stop()

    # Successful update.
    return optimizer.update_successful, grad_norm, num_zeros_in_grad

@torch.no_grad()
def optimizer_helper_bucket_wise_inner_step(optimizer, gbuf_index, dtype, bucket_index):
    """
    This bucket-wise optimizer step is only for twice-gradient-reduce.
    When twice gradient reduce is enabled, optimizer step will operate once for each iter. 
    First time using low precision, second time using higher precision.
    When gradient reduce overlapping is enabled, gradient buffer will be partitioned and form multiple buckets.
    
    Before low precision gradient reduce start for a bucket, the gradient should offloadded to CPU first.
    This will wait for last iter high precision reduce and optimizer step on high precision gradient finished.
    Thus, to avoid non necessary synchronization, optimizer step should operate on bucket-wise.

    Note: You need to pre check before excution. Since this will only triggered when last iter updated successfully. So no inf check and all gather inside.
    """

    assert isinstance(optimizer, DistributedOptimizer)
    
    for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
        bucket_shard_fp32_params_this_group = optimizer.bucket_wise_shard_fp32_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
        bucket_shard_fp32_from_float16_params_this_group = optimizer.bucket_wise_shard_fp32_from_float16_groups.get((gbuf_index, dtype, bucket_index, group_index), [])
        param_group['params'] = [
            *bucket_shard_fp32_params_this_group,
            *bucket_shard_fp32_from_float16_params_this_group,
        ]
    optimizer.optimizer.step()

    """ We need to reset param_group['step], since optimzier.step will increase step by inside, as a result param_group['step'] will increase multiple times (once for each bucket)."""
    for group_index, param_group in enumerate(optimizer.optimizer.param_groups):
        param_group['step'] -= 1

def debug_print_for_param_groups(param_groups):
    for group_index, group in enumerate(param_groups):
        orig_group = group["orig_group"]
        logging.debug(f"Group {group_index}:")
        logging.debug(f" group.keys(): {group.keys()}")
        logging.debug(f" orig_group.keys(): {orig_group.keys()}")
        logging.debug(f"orig_group['params'], val = ")
        for param in orig_group["params"]:
            logging.debug(f"    {param.shape}")
        logging.debug(f"key=lr, val = {orig_group['lr']}")
        logging.debug("")
    
def clip_grad_norm_fp32_with_pregiven_totalnorm(
    parameters,
    grads_for_norm,
    max_norm,
    check_for_nan_in_grad,
    norm_type=2,
    model_parallel_group=None,
    total_norm=None,
):
    """
    This clip grad norm is for bucket wise optimizer step, since for bucket wise optimizer step, total norm of all grads need to be given.
    Since this is only for Fast-Slow Gradient Reduce, so at second Gradient optimizer the total norm already known.
    """

    assert total_norm is not None, "total_norm need to be given"

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff
        )

    return total_norm

def calculate_pre_given_total_norm(
    parameters,
    grads_for_norm,
    max_norm,
    check_for_nan_in_grad,
    norm_type=2,
    model_parallel_group=None,
):
    """
    Unit Test only.
    This function used to calculate total norm.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Check individual rank grad norms are not NaN
        # prior to model-parallel all-reduce.
        if check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            assert not total_norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backwards pass. Device: {torch.cuda.current_device()}, '
                f'node: {os.uname()[1]}'
            )

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm