# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from logging import getLogger
from typing import Dict, List
from functools import reduce

import torch

from .. import parallel_state
from ...quantization_helper import QuantizationHelper

logger = getLogger(__name__)


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class StaleBucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Arguments:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.

        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        grad_reduce_stream: torch.cuda.Stream,
        use_distributed_optimizer: bool,
        fast_slow_grad_reduce_helper = None,
        parent_bucket = None,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        self.stale_data_buffer = None
        self.reduced_grads = None
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.use_distributed_optimizer = use_distributed_optimizer
        assert isinstance(grad_reduce_stream, torch.cuda.Stream)
        self.comm_stream = grad_reduce_stream
        self.fast_slow_grad_reduce_helper = fast_slow_grad_reduce_helper
        self.parent_bucket = parent_bucket
        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_event: torch.cuda.Event = None
        self.communication_issued = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        """
        assert (
            self.communication_event is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        stream = self.comm_stream
        
        event = torch.cuda.Event()
        self.communication_event = event
        self.communication_issued = True

        if self.use_distributed_optimizer:
            if self.data.device.type == 'cuda':
                # self.data /= self.data_parallel_world_size
                
                local_data_view = shard_buffer(self.data, self.data_parallel_world_size)[
                    self.data_parallel_rank
                ]

                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    torch.distributed._reduce_scatter_base(
                        local_data_view,
                        self.data,
                        group=self.data_parallel_group,
                        async_op=False,
                    )
                    # print(f'JINDA_DEBUG: StaleBucket start_grad_sync() GPU', flush=True)
                    # fast_slow_grad_reduce_helper.bucket_wise_copy_high_precision_grads_to_main_grads_each_bucket(parent_bucket)
                    # torch.cuda.synchronize()
                    # self.fast_slow_grad_reduce_helper.bucket_wise_optimizer_step(self.parent_bucket)
                    # self.reduced_grads = local_data_view.clone() # DEBUG_ONLY
                    event.record()
            else:
                # print('JINDA_DEBUG: StaleBucket start_grad_sync() CPU', flush=True)
                # self.data /= self.data_parallel_world_size
                # torch.cuda.synchronize()
                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    torch.distributed.all_reduce(
                        self.data,
                        group=self.data_parallel_group,
                        async_op=True,
                    )
                    event.record()

        else:
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(
                    self.data, group=self.data_parallel_group, async_op=False
                )
                event.record()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        """

        assert self.communication_event is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_event.wait()
        self.communication_event = None

    # def call_optimizer_func(self):
    #     self.optimizer_ref.copy_high_precision_grads_to_main_grads_each_bucket(self.gbuf_index, self.dtype, self.bucket_index)

class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Arguments:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        parent_ref,
        params: List[torch.nn.Parameter],
        data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        overlap_grad_reduce: bool,
        grad_reduce_stream: torch.cuda.Stream,
        use_distributed_optimizer: bool,
        quantization_helper: QuantizationHelper,
        fast_slow_grad_reduce_helper = None,
        DtoH_stream = None,
        HtoD_stream = None,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.parent_ref = parent_ref
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        self.stale_data_buffer = None
        self.last_iter_reduced_grads = None
        self.fast_slow_grad_reduce_helper = fast_slow_grad_reduce_helper
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.quantization_helper = quantization_helper
        self.stale_bucket = None
        if self.overlap_grad_reduce:
            assert isinstance(grad_reduce_stream, torch.cuda.Stream)
            self.comm_stream = grad_reduce_stream
        else:
            self.comm_stream = torch.cuda.default_stream()
        self.DtoH_stream = DtoH_stream
        self.HtoD_stream = HtoD_stream
        self.reset()
        self.handle_for_stale_bucket_copy = None
        self.bucket_wise_optimizer_event = None
    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_event: torch.cuda.Event = None
        self.communication_issued = False

    def set_stale_data_buffer(self, stale_bucket: StaleBucket):
        """
        Set stale data buffer for second time high precision gradient reduce-scatter.
        """
        self.stale_bucket = stale_bucket

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_event is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        stream = self.comm_stream
        # stale_handle_event = self.handle_for_stale_bucket_copy
        stale_handle_event = None
        event = torch.cuda.Event()
        self.communication_event = event
        self.communication_issued = True

        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
            local_data_view = shard_buffer(self.data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            if self.quantization_helper and self.quantization_helper.quantized_gradients:
                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    self.quantization_helper.quantize_reduce_gradients(self.data, local_data_view, stale_handle_event)
                    # self.last_iter_reduced_grads = local_data_view.clone() # DEBUG_ONLY
                    event.record()
            else:
                if stale_handle_event is not None:
                    stale_handle_event.wait()
                
                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    torch.distributed._reduce_scatter_base(
                        local_data_view,
                        self.data,
                        group=self.data_parallel_group,
                        async_op=False,
                    )
                    event.record()
        else:
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(
                    self.data, group=self.data_parallel_group, async_op=False
                )
                event.record()
        if not self.overlap_grad_reduce:
            self.communication_event.wait()
            self.communication_event = None

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_event is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_event.wait()
        self.communication_event = None
        self.data /= self.data_parallel_world_size

        # if self.fast_slow_grad_reduce_helper is not None:
        #     bucket_map_to_global_idx = self.fast_slow_grad_reduce_helper.optimizer.bucket_map_to_global_idx
            # (gbuf_index, dtype, bucket_index) = bucket_map_to_global_idx[self]
            # local_data_view = shard_buffer(self.data, self.data_parallel_world_size)[
            #     self.data_parallel_rank
            # ]
            # print(f'JINDA_DEBUG: low precision reduce reduce sync!!! gbuf_idx: {gbuf_index}, bucket_idx: {bucket_index}, stale_bucket_norm: {torch.norm(local_data_view)}', flush=True)

    def finish_stale_grad_sync(self):
        self.stale_bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        
        if len(self.params_with_grad) == 1:
            if self.stale_bucket is not None and self.stale_bucket.communication_issued:
                self.stale_bucket.finish_grad_sync()
                self.stale_bucket.reset()
                if self.fast_slow_grad_reduce_helper and self.fast_slow_grad_reduce_helper.last_iter_updated_successfully:
                    self.bucket_wise_optimizer_event = torch.cuda.Event()
                    self.HtoD_stream.wait_stream(torch.cuda.default_stream())
                    with torch.cuda.stream(self.HtoD_stream):
                        self.fast_slow_grad_reduce_helper.bucket_wise_copy_high_precision_grads_to_main_grads_each_bucket(self)
                        self.bucket_wise_optimizer_event.record()
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):

            if self.stale_bucket is not None:
                # if self.stale_bucket.communication_issued:
                #     bucket_map_to_global_idx = self.fast_slow_grad_reduce_helper.optimizer.bucket_map_to_global_idx
                #     (gbuf_index, dtype, bucket_index) = bucket_map_to_global_idx[self]
                #     self.stale_bucket.finish_grad_sync()
                #     self.stale_bucket.reset()
                #     if self.fast_slow_grad_reduce_helper and self.fast_slow_grad_reduce_helper.last_iter_updated_successfully:
                #         self.fast_slow_grad_reduce_helper.bucket_wise_copy_high_precision_grads_to_main_grads_each_bucket(self)
                #         self.fast_slow_grad_reduce_helper.bucket_wise_optimizer_step(self)
                
                if self.bucket_wise_optimizer_event is not None:
                    self.bucket_wise_optimizer_event.wait()
                    self.bucket_wise_optimizer_event = None
                if self.fast_slow_grad_reduce_helper and self.fast_slow_grad_reduce_helper.last_iter_updated_successfully:
                        self.fast_slow_grad_reduce_helper.bucket_wise_optimizer_step(self)
                        self.fast_slow_grad_reduce_helper.zero_optimizer_shard_grad()
                event = torch.cuda.Event()
                self.handle_for_stale_bucket_copy = event
                self.parent_ref.finsh_using_shared_temp_buffer()
                self.parent_ref.start_using_temp_buffer(event)
                temp_cuda_buffer = self.data.clone()
                self.DtoH_stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(self.DtoH_stream):
                    temp_cuda_buffer.div_(self.data_parallel_world_size)
                    self.stale_bucket.data.copy_(temp_cuda_buffer, non_blocking=True)
                    temp_cuda_buffer = None
                    event.record()
                # torch.cuda.synchronize()
                # bucket_map_to_global_idx = self.fast_slow_grad_reduce_helper.optimizer.bucket_map_to_global_idx
                # (gbuf_index, dtype, bucket_index) = bucket_map_to_global_idx[self]
                # print(f'JINDA_DEBUG: copy data to stale bucket, gbuf_idx: {gbuf_index}, bucket_idx: {bucket_index}, stale_bucket_norm: {torch.norm(self.stale_bucket.data)}', flush=True)
            self.start_grad_sync()
            # if self.stale_bucket is not None:
            #     # self.stale_bucket.data.copy_(self.data)
            #     self.stale_bucket.start_grad_sync()


class GradBuffer:
    """
    Groups gradients into a contiguous buffer, and then breaks the buffer into buckets with
    roughly `bucket_size` parameters each.

    Arguments:
        dtype: Type of underlying tensor.
        params: List of parameters whose gradients are collated in the underlying tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        quantization_helper = None,
        fast_slow_grad_reduce_helper = None,
    ):

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.dtype = dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.overlap_grad_reduce = overlap_grad_reduce
        self.quantization_helper = quantization_helper
        self.fast_slow_grad_reduce_helper = fast_slow_grad_reduce_helper
        self.grad_reduce_stream = None
        self.high_precision_slow_reduce_stream = None
        if self.overlap_grad_reduce:
            self.grad_reduce_stream = torch.cuda.Stream()
            if fast_slow_grad_reduce_helper is not None:
                self.high_precision_slow_reduce_stream = torch.cuda.Stream()
                # self.copy_grads_and_step_stream = torch.cuda.Stream()
                # self.copy_grads_and_step_event = None
        self.use_distributed_optimizer = use_distributed_optimizer
        self.is_last_microbatch = True
        # self.stale_grad_sync_issused = False

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.stale_buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        self.temp_buffer_handle = None

        def _pad_if_needed(data_index: int):
            bucket_size_divisible_by = 1
            if use_distributed_optimizer and quantization_helper is not None and (quantization_helper.quantized_weights or quantization_helper.quantized_gradients):
                def least_common_multiple(divisors):
                    """Find least common multiple of a list of numbers."""
                    lcm_value = reduce(lambda x, y: x * y // math.gcd(x, y), divisors)
                    return lcm_value
                weight_quantization_pad = 1
                gradient_quantization_pad = 1
                if quantization_helper.quantized_weights:
                    weight_quantization_pad = quantization_helper.wq_group_size
                if quantization_helper.quantized_gradients:
                    gradient_quantization_pad = least_common_multiple([quantization_helper.gq_group_size_intra * quantization_helper.gradient_alltoall_pipeline, quantization_helper.gq_group_size_inter])
                bucket_size_divisible_by = least_common_multiple([weight_quantization_pad, gradient_quantization_pad]) * self.data_parallel_world_size
                return (
                    int(math.ceil(data_index / bucket_size_divisible_by))
                    * bucket_size_divisible_by
                )
            """Pads data indices if using distributed optimizer (to ensure uniform sharding)."""
            if use_distributed_optimizer:
                return (
                    int(math.ceil(data_index / self.data_parallel_world_size))
                    * self.data_parallel_world_size
                )
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0
        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already, form a new bucket.
            # If bucket_size is None, accumulate everything into a single bucket.

            # TODO: Remove len(bucket_params) > 1 when the final head that transforms token
            # representations from hidden space to vocabulary space is in a PyTorch module
            # whose forward method is called. If it is not and a bucket contains only this
            # one parameter, we get incorrect behavior (i.e., higher losses) since we do not
            # call the wait function on the bucket's all_gather_handle (we use forward pre-
            # hooks on PyTorch modules to do this when --overlap-param-gather is used).
            # As a temporary workaround, we make sure that no bucket has only one parameter.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size and len(
                    bucket_params
                ) > 1:
                    per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
                    data_end_index = _pad_if_needed(data_end_index)
                    self.bucket_indices.append((bucket_data_start_index, data_end_index))
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
                    bucket_id += 1
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
            data_end_index = _pad_if_needed(data_end_index)
            self.bucket_indices.append((bucket_data_start_index, data_end_index))

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        if use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        self.data = torch.zeros(
            self.numel, dtype=self.dtype, device=torch.cuda.current_device(), requires_grad=False,
        )
        
        self.grad_for_high_precision_reduce_data = None
        if self.fast_slow_grad_reduce_helper is not None:
            device = 'cpu'
            if self.fast_slow_grad_reduce_helper.high_precision_grad_device == 'cuda':
                device = 'cuda'
            self.grad_for_high_precision_reduce_data = torch.empty(
                self.numel, 
                dtype=self.dtype, 
                device=device, 
                requires_grad=False, 
                pin_memory=True
            )
        # torch.device('cpu')
        
        DtoH_stream = None
        HtoD_stream = None
        if self.fast_slow_grad_reduce_helper is not None:
            DtoH_stream = torch.cuda.Stream()
            HtoD_stream = torch.cuda.Stream()
        # Finally, map main_grad fields for each parameter with a .grad field.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param.main_grad = self._get(param.data.shape, data_start_index)
            if self.fast_slow_grad_reduce_helper is not None:
                param.stale_grad = self._get(param.data.shape, data_start_index, grad_for_high_precision_reduce=True)
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                    DtoH_stream=DtoH_stream,
                    HtoD_stream=HtoD_stream,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params=bucket_params,
                start_index=bucket_data_start_index,
                end_index=bucket_data_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
                DtoH_stream=DtoH_stream,
                HtoD_stream=HtoD_stream,
            )

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')

    def _get(self, shape: torch.Size, start_index: int, grad_for_high_precision_reduce=False) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        if grad_for_high_precision_reduce is False:
            buffer_tensor = self.data[start_index:end_index]
        else:
            buffer_tensor = self.grad_for_high_precision_reduce_data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
        DtoH_stream = None,
        HtoD_stream = None,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]



        # Get appropriate view into global GradBuffer.
        bucket_data = self._get(torch.Size([end_index - start_index]), start_index, grad_for_high_precision_reduce=False)
        bucket = Bucket(
            self,
            params=bucket_params,
            data=bucket_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            overlap_grad_reduce=self.overlap_grad_reduce,
            grad_reduce_stream=self.grad_reduce_stream,
            use_distributed_optimizer=self.use_distributed_optimizer,
            quantization_helper=self.quantization_helper,
            fast_slow_grad_reduce_helper=self.fast_slow_grad_reduce_helper,
            DtoH_stream=DtoH_stream,
            HtoD_stream=HtoD_stream,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket
        if self.fast_slow_grad_reduce_helper is not None:
            high_precision_grad_data_buffer = self._get(torch.Size([end_index - start_index]), start_index, grad_for_high_precision_reduce=True)
            high_precision_grad_bucket = StaleBucket(
                params=bucket_params,
                data=high_precision_grad_data_buffer,
                offset=start_index,
                numel_unpadded=numel_unpadded,
                data_parallel_group=self.data_parallel_group,
                data_parallel_world_size=self.data_parallel_world_size,
                grad_reduce_stream=self.high_precision_slow_reduce_stream,
                use_distributed_optimizer=self.use_distributed_optimizer,
                fast_slow_grad_reduce_helper=self.fast_slow_grad_reduce_helper,
                parent_bucket=bucket
            )
            bucket.set_stale_data_buffer(high_precision_grad_bucket)
            self.stale_buckets.append(high_precision_grad_bucket)

    def reset(self, zero_buffer):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.

        When zero_buffer is set to True, the underlying buffer is zeroed out.
        """
        if zero_buffer:
            self.data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def reset_stale_buffer(self, zero_buffer):
        if zero_buffer:
            self.grad_for_high_precision_reduce_data.zero_()
        for bucket in self.stale_buckets:
            bucket.reset()

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def start_stale_grad_sync(self):
        # if torch.distributed.get_rank() == 0:
        #     print('GradBuffer start stale grad sync', flush=True)
        if self.fast_slow_grad_reduce_helper and self.fast_slow_grad_reduce_helper.last_iter_updated_successfully:
            # print("JINDA_DEBUG: start_stale_grad_sync() last_iter_updated_successfully is True", flush=True)
            for stale_bucket in self.stale_buckets:
                """Before starting high precision gradient reduce-scatter, we need to assure
                that D2H copy of the high precision gradients is finished. This is because we
                need to offload high precision gradients to CPU, to reduce the GPU memory usage."""
                if stale_bucket.parent_bucket.handle_for_stale_bucket_copy is not None:
                    stale_bucket.parent_bucket.handle_for_stale_bucket_copy.synchronize()
                    stale_bucket.parent_bucket.handle_for_stale_bucket_copy = None
                stale_bucket.start_grad_sync()

    # def finish_stale_grad_sync(self):
    #     """
    #     Finishes grad sync (all-reduce or reduce-scatter) communication operations
    #     for all buckets in the grad buffer.

    #     When overlap_grad_reduce is set to True, waits for asynchronous communication
    #     calls to complete. When overlap_grad_reduce is set to False, calls synchronous
    #     communication ops.
    #     """
    #     # if torch.distributed.get_rank() == 0:
    #     #     print('GradBuffer finish stale grad sync', flush=True)
    #     assert self.stale_grad_sync_issused, 'stale_grad_sync_issused should be True'
    #     self.stale_grad_sync_issused = False
    #     for stale_bucket in self.stale_buckets:
    #         stale_bucket.finish_grad_sync()
    #         stale_bucket.reset()
        
    # def finish_copy_stale_grads_and_step(self):
    #     if self.copy_grads_and_step_event is not None:
    #         self.copy_grads_and_step_event.wait()
    #         self.copy_grads_and_step_event = None
    
    # def copy_grads_and_step(self):    
    #     stream = self.copy_grads_and_step_stream
    #     event = torch.cuda.Event()
    #     self.copy_grads_and_step_event = event
    #     stream.wait_stream(torch.cuda.default_stream())
    #     with torch.cuda.stream(stream):
    #         for bucket in self.buckets:
    #             self.fast_slow_grad_reduce_helper.bucket_wise_copy_high_precision_grads_to_main_grads_each_bucket(bucket)
    #             self.fast_slow_grad_reduce_helper.bucket_wise_optimizer_step(bucket)
    #         event.record()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
    
    def start_using_temp_buffer(self, event):
        """Since we need to offload high precision gradients to CPU, and this transfer is slow, we need to use a temp buffer 
        to store the high precision gradients before offloading them to CPU. To reduce the GPU memory peak usage, at most one temp
        buffer is used at a time. This function is used to indicate that the temp buffer is in use.
        """
        assert self.temp_buffer_handle is None, 'temp_buffer_handle already in used, please wait until it finished'
        self.temp_buffer_handle = event

    def finsh_using_shared_temp_buffer(self):
        if self.temp_buffer_handle is not None:
            self.temp_buffer_handle.synchronize()
            self.temp_buffer_handle = None
    