import torch
import sys
import megatron.core.parallel_state as ps
from megatron.timers import Timers
from tests.unit_tests.test_utilities import Utils
sys.path.append('/ocean/projects/asc200010p/jjia1/scripts/analysis')
import os 
from megatron.core.tensor_parallel.mappings import _reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim, _gather_along_last_dim, _reduce_scatter_along_last_dim, all_to_all_reduce_scatter_along_first_dim, _all_to_all_along_first_dim
from compression.compress import get_any_comp_timings, get_float_comp_timings, calc_comp_ratio, compress_data, decompress_data, max_any_compressed_output_size
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_diff
from torch.quantization.observer import MovingAverageMinMaxObserver, MinMaxObserver
import numpy as np
import fbgemm_gpu

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

_GLOBAL_TIMERS = None

def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)

def print_all(msg):
    msgs = [None for _ in range(torch.cuda.device_count())]
    msg = ('rank {}'.format(torch.distributed.get_rank()), msg)
    torch.distributed.all_gather_object(msgs,
                                        msg)
    if torch.distributed.get_rank() == 0:
        for msg in msgs:
            print(msg)

def test_all_reduce_tensor_group(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing all_reduce_model_parallel with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # timers = get_timers()
    copy_input = input.clone()
    """warm up"""
    for i in range(5):
        _reduce(input)
        input.copy_(copy_input)

    """start test"""
    # timers('all-reduce', log_level=2).start()
    start.record()
    _reduce(input)
    end.record()
    torch.cuda.synchronize()
    print_rank_0('cuda timer: all-reduce: {}ms'.format(start.elapsed_time(end)))
    # timers('all-reduce').stop()
    # timers.log(['all-reduce'], barrier=True)
    return input

def test_reduce_scatter_all_gather_along_first_dim(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing reduce_scatter_all_gather_along_first_dim, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    # timers = get_timers()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    temp_start = torch.cuda.Event(enable_timing=True)
    temp_end = torch.cuda.Event(enable_timing=True)   
    
    """initizalize buffer"""
    dim_size = list(input.size())
    assert (
        dim_size[0] % tensor_model_parallel_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[0] = dim_size[0] // tensor_model_parallel_size
    reduce_scatter_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
    dim_size = list(input.size())
    gather_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())

    """warm up"""
    for i in range(5):
        _reduce_scatter_along_first_dim(input, reduce_scatter_output)
        _gather_along_first_dim(reduce_scatter_output, gather_output)

    """start test"""
    start.record()

    temp_start.record()
    _reduce_scatter_along_first_dim(input, reduce_scatter_output)
    temp_end.record()
    torch.cuda.synchronize()
    reduce_scatter_time = temp_start.elapsed_time(temp_end)

    temp_start.record()
    _gather_along_first_dim(reduce_scatter_output, gather_output)
    temp_end.record()
    torch.cuda.synchronize()
    all_gather_time = temp_start.elapsed_time(temp_end)

    end.record()
    torch.cuda.synchronize()

    print_rank_0('cuda timer: reduce_scatter: {}ms'.format(reduce_scatter_time))
    print_rank_0('cuda timer: all gather: {}ms'.format(all_gather_time))
    print_rank_0('cuda timer: reduce_scatter_all_gather: {}ms'.format(start.elapsed_time(end)))

    return gather_output

def test_all2all_all_gather_along_first_dim(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing test_all2all_all_gather_along_first_dim, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    """initizalize buffer"""
    dim_size = list(input.size())
    all_to_all_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
    gather_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())

    timers = get_timers()
    # output = all_to_all_reduce_scatter_along_first_dim(input)
    # gather_tensor_list = [torch.empty(list(output.shape), dtype=input.dtype, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    # final_output = torch.empty(list(input.shape), dtype=input.dtype, device=torch.cuda.current_device())

    """warm up"""
    for i in range(5):
        _all_to_all_along_first_dim(input, all_to_all_output)
        chunks = all_to_all_output.chunk(tensor_model_parallel_size)
        summed_tensor = torch.stack(chunks).sum(dim=0)
        _gather_along_first_dim(summed_tensor, gather_output)

    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    _all_to_all_along_first_dim(input, all_to_all_output)
    timers('reduce_scatter').stop()
    # print('output: {}, rank: {}'.format(output, torch.distributed.get_rank()))
    timers('tensor-sum', log_level=2).start()
    chunks = all_to_all_output.chunk(tensor_model_parallel_size)
    summed_tensor = torch.stack(chunks).sum(dim=0)
    timers('tensor-sum').stop()

    timers('all_gather', log_level=2).start()
    _gather_along_first_dim(summed_tensor, gather_output)
    timers('all_gather').stop()
    # print('gather_tensor_list: {}, rank: {}'.format(gather_tensor_list, torch.distributed.get_rank()))

    timers('reduce_scatter_all_gather').stop()
    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gather', 'tensor-sum'], barrier=True)
    return gather_output

def test_all2all_all_quantization(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing test_all2all_all_quantization, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    timers = get_timers()

    """allocate buffer"""
    timers('allocate_output_buffer', log_level=2).start()

    """all to all reduce scatter buffer"""
    dim_size = list(input.size())
    assert (
        dim_size[0] % get_tensor_model_parallel_world_size() == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    scatter_size_per_dim = dim_size[0] // get_tensor_model_parallel_world_size()
    dim_size[0] = scatter_size_per_dim
    input_split = list(input.split(scatter_size_per_dim, dim=0))

    row_dim = 4096
    input_2d = input_split[0].view((-1, row_dim)) if row_dim > 0 else input_split[0]
    quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    input_quant_split = [torch.empty_like(quantized_tensor, dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    output_quant_list = [torch.empty_like(quantized_tensor, dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    assert (
        dim_size[0] % get_tensor_model_parallel_world_size() == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[0] = dim_size[0] // get_tensor_model_parallel_world_size()

    temp_dequantized = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(quantized_tensor, 4)
    reduce_scatter_output = torch.empty_like(temp_dequantized, dtype=torch.half, device=torch.cuda.current_device())

    """all gather buffer"""
    output = _reduce_scatter_along_first_dim(input)
    original_shape = output.shape
    row_dim = 4096
    input_2d = output.view((-1, row_dim)) if row_dim > 0 else output
    quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    buffer_shape = quantized_tensor.shape
    quantized_tensor_list = [torch.empty((tuple(buffer_shape)), dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    dequantized_output = torch.empty(dim_size, dtype=torch.half, device=torch.cuda.current_device())

    timers('allocate_output_buffer').stop()

    """warm up"""
    for i in range(5):
        input_split = list(input.split(scatter_size_per_dim, dim=0))
        for j in range(len(input_split)):
            input_2d = input_split[j].view((-1, row_dim)) if row_dim > 0 else input_split[j]
            temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
            input_quant_split[j] = temp_quantized_tensor
        torch.distributed.all_to_all(output_quant_list, input_quant_split, group=get_tensor_model_parallel_group())
        reduce_scatter_output.zero_()
        for i in range(len(output_quant_list)):
            temp = output_quant_list[i]
            temp_dequantized = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(temp, 4)
            reduce_scatter_output = reduce_scatter_output + temp_dequantized
        """all gather"""
        input_2d = reduce_scatter_output.view((-1, row_dim)) if row_dim > 0 else reduce_scatter_output
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
        torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
        index = 0
        for quantized_tensor in quantized_tensor_list:
            dequantized_tensor = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(quantized_tensor, 4)
            dequantized_tensor = dequantized_tensor.view(original_shape)
            end = index + dequantized_tensor.size(0)
            dequantized_output[index:end] = dequantized_tensor
            index += dequantized_tensor.size(0)

    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('all-to-all-quantize', log_level=2).start()
    input_split = list(input.split(scatter_size_per_dim, dim=0))
    for j in range(len(input_split)):
        input_2d = input_split[j].view((-1, row_dim)) if row_dim > 0 else input_split[j]
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
        input_quant_split[j] = temp_quantized_tensor
    timers('all-to-all-quantize').stop()

    timers('all-to-all', log_level=2).start()
    torch.distributed.all_to_all(output_quant_list, input_quant_split, group=get_tensor_model_parallel_group())
    timers('all-to-all').stop()

    timers('all-to-all-dequantize-reduce', log_level=2).start()
    reduce_scatter_output.zero_()
    for i in range(len(output_quant_list)):
        temp = output_quant_list[i]
        temp_dequantized = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(temp, 4)
        reduce_scatter_output = reduce_scatter_output + temp_dequantized
    timers('all-to-all-dequantize-reduce').stop()

    timers('all_gather-quantize', log_level=2).start()
    input_2d = reduce_scatter_output.view((-1, row_dim)) if row_dim > 0 else reduce_scatter_output
    temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    timers('all_gather-quantize').stop()

    timers('all_gather', log_level=2).start()
    torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
    timers('all_gather').stop()

    timers('concat', log_level=2).start()
    index = 0
    for quantized_tensor in quantized_tensor_list:
        dequantized_tensor = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(quantized_tensor, 4)
        dequantized_tensor = dequantized_tensor.view(original_shape)
        end = index + dequantized_tensor.size(0)
        dequantized_output[index:end] = dequantized_tensor
        index += dequantized_tensor.size(0)
    timers('concat').stop()

    timers('reduce_scatter_all_gather').stop()
    timers.log(['reduce_scatter_all_gather', 'all-to-all-quantize',  'all-to-all',  'all-to-all-dequantize-reduce', 'all_gather-quantize', 'all_gather', 'concat'], barrier=True)
    return dequantized_output


def test_reduce_scatter_all_gather_along_last_dim(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing reduce_scatter_all_gather_along_last_dim, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    timers = get_timers()
    
    """warm up"""
    for i in range(5):
        output = _reduce_scatter_along_last_dim(input)
        output = _gather_along_last_dim(output)

    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    output = _reduce_scatter_along_last_dim(input)
    timers('reduce_scatter').stop()

    timers('all_gather', log_level=2).start()
    output = _gather_along_last_dim(output)
    timers('all_gather').stop()

    timers('reduce_scatter_all_gather').stop()
    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gather'], barrier=True)
    return output

def test_reduce_scatter_all_gather_quantization_per_token_int4(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> test_reduce_scatter_all_gather_quantization_per_token_int4, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))
    
    timers = get_timers()
    
    """allocate quantized_tensor_list"""
    timers('allocate_output_buffer', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    original_shape = output.shape
    row_dim = 4096
    input_2d = output.view((-1, row_dim)) if row_dim > 0 else output
    quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    buffer_shape = quantized_tensor.shape
    quantized_tensor_list = [torch.empty((tuple(buffer_shape)), dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    dequantized_output = torch.empty(dim_size, dtype=torch.half, device=torch.cuda.current_device())
    timers('allocate_output_buffer').stop()

    """warm up"""
    for _ in range(5):
        temp_output = _reduce_scatter_along_first_dim(input)
        input_2d = temp_output.view((-1, row_dim)) if row_dim > 0 else temp_output
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
        torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
        index = 0
        for quantized_tensor in quantized_tensor_list:
            dequantized_tensor = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(quantized_tensor, 4)
            dequantized_tensor = dequantized_tensor.view(original_shape)
            end = index + dequantized_tensor.size(0)
            dequantized_output[index:end] = dequantized_tensor
            index += dequantized_tensor.size(0)

    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    timers('reduce_scatter').stop()

    """start quantizing"""
    timers('quantizing', log_level=2).start()
    input_2d = output.view((-1, row_dim)) if row_dim > 0 else output
    quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    timers('quantizing').stop()

    """start all gather"""
    timers('all_gather', log_level=2).start()
    torch.distributed.all_gather(quantized_tensor_list, quantized_tensor, group=get_tensor_model_parallel_group())
    timers('all_gather').stop()

    """start dequantize"""
    timers('dequantize', log_level=2).start()
    index = 0
    for quantized_tensor in quantized_tensor_list:
        dequantized_tensor = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(quantized_tensor, 4)
        dequantized_tensor = dequantized_tensor.view(original_shape)
        end = index + dequantized_tensor.size(0)
        dequantized_output[index:end] = dequantized_tensor
        index += dequantized_tensor.size(0)
    timers('dequantize').stop()
    
    timers('reduce_scatter_all_gather').stop()

    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gather', 'allocate_output_buffer', 'quantizing', 'dequantize'], barrier=True)
    print('dequantized_output.shape: {}'.format(dequantized_output.shape))

    return dequantized_output

def test_reduce_scatter_all_gather_quantization_per_token_int8(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> test_reduce_scatter_all_gather_quantization_per_token_int8, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))
    
    timers = get_timers()
    
    """allocate quantized_tensor_list"""
    timers('allocate_output_buffer', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    original_shape = output.shape
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(output.contiguous())
    buffer_shape = quantized_tensor.shape
    quantized_tensor_list = [torch.empty((1, *tuple(buffer_shape)), dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    dequantized_output = torch.empty(dim_size, dtype=torch.half, device=torch.cuda.current_device())
    timers('allocate_output_buffer').stop()

    """warm up"""
    for _ in range(5):
        temp_output = _reduce_scatter_along_first_dim(input)
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(temp_output.contiguous())
        torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
        
        index = 0
        for quantized_tensor in quantized_tensor_list:
            dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
            dequantized_tensor = dequantized_tensor.view(original_shape)
            end = index + dequantized_tensor.size(0)
            dequantized_output[index:end] = dequantized_tensor
            index += dequantized_tensor.size(0)


    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    timers('reduce_scatter').stop()

    """start quantizing"""
    timers('quantizing', log_level=2).start()
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(output.contiguous())    
    timers('quantizing').stop()

    """start all gather"""
    timers('all_gather', log_level=2).start()
    torch.distributed.all_gather(quantized_tensor_list, quantized_tensor, group=get_tensor_model_parallel_group())
    timers('all_gather').stop()

    """start dequantize"""
    timers('dequantize', log_level=2).start()
    index = 0
    for quantized_tensor in quantized_tensor_list:
        dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
        dequantized_tensor = dequantized_tensor.view(original_shape)
        end = index + dequantized_tensor.size(0)
        dequantized_output[index:end] = dequantized_tensor
        index += dequantized_tensor.size(0)
    timers('dequantize').stop()
    
    timers('reduce_scatter_all_gather').stop()

    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gather', 'allocate_output_buffer', 'quantizing', 'dequantize'], barrier=True)
    print('dequantized_output.shape: {}'.format(dequantized_output.shape))

    return dequantized_output


def test_reduce_scatter_all_gather_compression_per_token_int4(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> test_reduce_scatter_all_gather_compression_per_token_int4, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    timers = get_timers()
    dev = torch.cuda.current_device()

    """buffer allocate"""
    timers('allocate_output_buffer', log_level=2).start()
    """allocate temp buffer"""
    gathered_tensor_lens = [torch.zeros([1], dtype=torch.int64, device=dev) for _ in range(get_tensor_model_parallel_world_size())]
    """allocate quantized_tensor_list"""
    output = _reduce_scatter_along_first_dim(input)
    row_dim = 4096
    original_shape = output.shape
    input_2d = output.view((-1, row_dim)) if row_dim > 0 else output
    quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    buffer_shape = quantized_tensor.shape
    quantized_tensor_list = [torch.empty((1, *tuple(buffer_shape)), dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    dequantized_output = torch.empty(dim_size, dtype=torch.half, device=torch.cuda.current_device())
    """ANS compress allocate buffer"""
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)
    ts_in = [quantized_tensor]
    rows, cols = max_any_compressed_output_size(ts_in)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts_in)], dtype=torch.int, device=dev)
    max_tensor_len = rows * cols
    tensor_list = [torch.empty((1, max_tensor_len), dtype=torch.uint8, device=dev) for _ in range(get_tensor_model_parallel_world_size())]

    timers('allocate_output_buffer').stop()


    
    """warm up"""
    for i in range(5):
        temp_output = _reduce_scatter_along_first_dim(input)
        input_2d = temp_output.view((-1, row_dim)) if row_dim > 0 else temp_output
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
        if i==0:
            print('quantize comression sizes: {}'.format(temp_quantized_tensor.shape))
        """ans compress data"""
        ts_in = [temp_quantized_tensor]
        compress_data(False, ts_in, False, tempMem, comp, sizes)
        if i==0:
            print('ANS comression sizes: {}'.format(sizes[0]))
        tensor_len_tensor = torch.tensor([sizes[0]], dtype=torch.int64, device=dev)
        torch.distributed.all_gather(gathered_tensor_lens, tensor_len_tensor, group=get_tensor_model_parallel_group())
        """create all gather output buffer with tensor_shape_list"""
        for idx, length_tensor in enumerate(gathered_tensor_lens):
                length = length_tensor.item()
                tensor_list[idx] = tensor_list[idx][:, :length]
        """all gather tensor with different length"""
        tcomp = comp[:, :sizes[0]]
        torch.distributed.all_gather(tensor_list, tcomp, group=get_tensor_model_parallel_group())

        # torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
        # index = 0
        # for quantized_tensor in quantized_tensor_list:
        #     dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
        #     dequantized_tensor = dequantized_tensor.view(original_shape)
        #     end = index + dequantized_tensor.size(0)
        #     dequantized_output[index:end] = dequantized_tensor


    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    timers('reduce_scatter').stop()

    """start quantizing"""
    timers('quantizing', log_level=2).start()
    input_2d = temp_output.view((-1, row_dim)) if row_dim > 0 else input
    temp_quantized_tensor = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_2d.contiguous(), 4)
    timers('quantizing').stop()

    timers('ans_compression', log_level=2).start()
    """ANS COMPRESSION START"""
    ts_in = [temp_quantized_tensor]
    compress_data(False, ts_in, False, tempMem, comp, sizes)
    timers('ans_compression').stop()

    timers('all_gather_tensor_shape', log_level=2).start()
    """all gather tensor shape"""
    tensor_len_tensor = torch.tensor([sizes[0]], dtype=torch.int64, device=dev)
    torch.distributed.all_gather(gathered_tensor_lens, tensor_len_tensor, group=get_tensor_model_parallel_group())
    timers('all_gather_tensor_shape').stop()

    timers('allocate_output_buffer', log_level=2).start()
    """create all gather output buffer with tensor_shape_list"""
    for idx, length_tensor in enumerate(gathered_tensor_lens):
            length = length_tensor.item()
            tensor_list[idx] = tensor_list[idx][:, :length]
    timers('allocate_output_buffer').stop()

    timers('all_gathe_tensor', log_level=2).start()
    comp = comp[:, :sizes[0]]
    torch.distributed.all_gather(tensor_list, comp, group=get_tensor_model_parallel_group())
    timers('all_gathe_tensor').stop()

    timers('reduce_scatter_all_gather').stop()
    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gathe_tensor', 'ans_compression', 'all_gather_tensor_shape', 'allocate_output_buffer', 'quantizing', 'quantizing-obs', 'quantizing-transaction'], barrier=True)
    return output

def test_reduce_scatter_all_gather_compression_per_token_int8(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> test_reduce_scatter_all_gather_compression_per_token_int8, with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))

    timers = get_timers()
    dev = torch.cuda.current_device()

    """buffer allocate"""
    timers('allocate_output_buffer', log_level=2).start()
    """allocate temp buffer"""
    gathered_tensor_lens = [torch.zeros([1], dtype=torch.int64, device=dev) for _ in range(get_tensor_model_parallel_world_size())]
    """allocate quantized_tensor_list"""
    output = _reduce_scatter_along_first_dim(input)
    original_shape = output.shape
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(output.contiguous())
    buffer_shape = quantized_tensor.shape
    quantized_tensor_list = [torch.empty((1, *tuple(buffer_shape)), dtype=torch.uint8, device=torch.cuda.current_device()) for _ in range(get_tensor_model_parallel_world_size())]
    dim_size = list(input.size())
    dequantized_output = torch.empty(dim_size, dtype=torch.half, device=torch.cuda.current_device())
    """ANS compress allocate buffer"""
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)
    ts_in = [quantized_tensor]
    rows, cols = max_any_compressed_output_size(ts_in)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts_in)], dtype=torch.int, device=dev)
    max_tensor_len = rows * cols
    tensor_list = [torch.empty((1, max_tensor_len), dtype=torch.uint8, device=dev) for _ in range(get_tensor_model_parallel_world_size())]

    timers('allocate_output_buffer').stop()


    
    """warm up"""
    for i in range(5):
        temp_output = _reduce_scatter_along_first_dim(input)
        temp_quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(temp_output.contiguous())
        if i==0:
            print('quantize comression sizes: {}'.format(temp_quantized_tensor.shape))
        """ans compress data"""
        ts_in = [temp_quantized_tensor]
        compress_data(False, ts_in, False, tempMem, comp, sizes)
        if i==0:
            print('ANS comression sizes: {}'.format(sizes[0]))
        tensor_len_tensor = torch.tensor([sizes[0]], dtype=torch.int64, device=dev)
        torch.distributed.all_gather(gathered_tensor_lens, tensor_len_tensor, group=get_tensor_model_parallel_group())
        """create all gather output buffer with tensor_shape_list"""
        for idx, length_tensor in enumerate(gathered_tensor_lens):
                length = length_tensor.item()
                tensor_list[idx] = tensor_list[idx][:, :length]
        """all gather tensor with different length"""
        tcomp = comp[:, :sizes[0]]
        torch.distributed.all_gather(tensor_list, tcomp, group=get_tensor_model_parallel_group())

        # torch.distributed.all_gather(quantized_tensor_list, temp_quantized_tensor, group=get_tensor_model_parallel_group())
        # index = 0
        # for quantized_tensor in quantized_tensor_list:
        #     dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
        #     dequantized_tensor = dequantized_tensor.view(original_shape)
        #     end = index + dequantized_tensor.size(0)
        #     dequantized_output[index:end] = dequantized_tensor


    """start test"""
    timers('reduce_scatter_all_gather', log_level=2).start()

    timers('reduce_scatter', log_level=2).start()
    output = _reduce_scatter_along_first_dim(input)
    timers('reduce_scatter').stop()

    """start quantizing"""
    timers('quantizing', log_level=2).start()
    temp_quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(temp_output.contiguous())
    timers('quantizing').stop()

    timers('ans_compression', log_level=2).start()
    """ANS COMPRESSION START"""
    ts_in = [temp_quantized_tensor]
    compress_data(False, ts_in, False, tempMem, comp, sizes)
    timers('ans_compression').stop()

    timers('all_gather_tensor_shape', log_level=2).start()
    """all gather tensor shape"""
    tensor_len_tensor = torch.tensor([sizes[0]], dtype=torch.int64, device=dev)
    torch.distributed.all_gather(gathered_tensor_lens, tensor_len_tensor, group=get_tensor_model_parallel_group())
    timers('all_gather_tensor_shape').stop()

    timers('allocate_output_buffer', log_level=2).start()
    """create all gather output buffer with tensor_shape_list"""
    for idx, length_tensor in enumerate(gathered_tensor_lens):
            length = length_tensor.item()
            tensor_list[idx] = tensor_list[idx][:, :length]
    timers('allocate_output_buffer').stop()

    timers('all_gathe_tensor', log_level=2).start()
    comp = comp[:, :sizes[0]]
    torch.distributed.all_gather(tensor_list, comp, group=get_tensor_model_parallel_group())
    timers('all_gathe_tensor').stop()

    timers('reduce_scatter_all_gather').stop()
    timers.log(['reduce_scatter_all_gather', 'reduce_scatter', 'all_gathe_tensor', 'ans_compression', 'all_gather_tensor_shape', 'allocate_output_buffer', 'quantizing', 'quantizing-obs', 'quantizing-transaction'], barrier=True)
    return output

def test_quantized_all_reduce_tensor_group(tensor_model_parallel_size, input):
    if torch.distributed.get_rank() == 0:
        print('> testing all_reduce_model_parallel with parallel size {} data size {} ...'.format(
            tensor_model_parallel_size, input.shape))
    
    timers = get_timers()
    copy_input = input.clone()
    """warm up"""
    for i in range(5):
        quantized_input = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input.contiguous())
        _reduce(quantized_input)
        input.copy_(copy_input)

    """start test"""
    timers('all-reduce', log_level=2).start()
    quantized_input = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input.contiguous())
    print_rank_0('quantized input type: {} shape: {}'.format(quantized_input.dtype, quantized_input.shape))
    _reduce(quantized_input)
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_input)
    timers('all-reduce').stop()
    timers.log(['all-reduce'], barrier=True)
    return dequantized_tensor

def get_timers():
    """Return timers."""
    assert _GLOBAL_TIMERS is not None, '{} is not initialized.'.format('timers')
    return _GLOBAL_TIMERS

def setup_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    assert _GLOBAL_TIMERS is None, '{} is already initialized.'.format('timers')
    _GLOBAL_TIMERS = Timers(2, 'all')

def data_provider(dev=0):
    tensor_rank = 'tensor_rank_{:d}.pt'.format(dev)

    """import origin tensor"""
    tensor_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00600/layer_001/ParallelMLP'
    tensor_path = os.path.join(tensor_path, tensor_rank)
    tensor = torch.load(tensor_path, map_location=torch.device(device=dev))
    return [tensor]

def time_test():
    distributed_env_inatialize()
    world_size = Utils.world_size

    # for i in range(5):
        # torch_shape = (2, 1024 * 2 ** i, 2304)
    # torch_shape = (4, 1536, 4096)
    """create random tensor"""
    # torch_shape = (4,2)
    # torch_shape = (4, 1536, 4096)
    # input = torch.empty(torch_shape, dtype=torch.float32, device=torch.cuda.current_device())

    input = data_provider(torch.cuda.current_device())[0]
    print_rank_0("input tensor shape: {}".format(input.shape))

    output = test_all_reduce_tensor_group(world_size, input.clone())

    temp_output = test_reduce_scatter_all_gather_along_first_dim(world_size, input.clone())
    analysis_diff(output, temp_output)

    temp_output = test_reduce_scatter_all_gather_along_last_dim(world_size, input.clone())
    analysis_diff(output, temp_output)

    temp_output = test_reduce_scatter_all_gather_quantization_per_token_int8(world_size, input.clone())
    analysis_diff(output, temp_output)

    test_reduce_scatter_all_gather_compression_per_token_int8(world_size, input.clone())

    temp_output = test_reduce_scatter_all_gather_quantization_per_token_int4(world_size, input.clone())
    analysis_diff(output, temp_output)

    test_reduce_scatter_all_gather_compression_per_token_int4(world_size, input.clone())

    temp_output = test_all2all_all_gather_along_first_dim(world_size, input.clone())
    analysis_diff(output, temp_output)

    temp_output = test_all2all_all_quantization(world_size, input.clone())
    analysis_diff(output, temp_output)

def debug_test():
    distributed_env_inatialize()
    world_size = Utils.world_size
    # input = torch.arange(4, dtype=torch.int64, device=torch.cuda.current_device()) + 1 + 2 * torch.distributed.get_rank()
    input = data_provider(torch.cuda.current_device())[0]
    # print('input: {}, rank: {}'.format(input, torch.distributed.get_rank()))
    output = test_all_reduce_tensor_group(world_size, input.clone())
    # print_rank_0('all-reduce: {}'.format(output))

    temp_output = test_reduce_scatter_all_gather_along_first_dim(world_size, input.clone())
    # print_rank_0('temp_output: {}'.format(temp_output))
    print_rank_0(analysis_diff(output, temp_output))

    temp_output = test_all2all_all_gather_along_first_dim(world_size, input.clone())
    print_rank_0(analysis_diff(output, temp_output))

def error_test():
    """initialize distributed environment"""
    distributed_env_inatialize()
    world_size = Utils.world_size

    # for i in range(5):
        # torch_shape = (2, 1024 * 2 ** i, 2304)
    # torch_shape = (4, 1536, 4096)
    """create random tensor"""
    # torch_shape = (4,2)
    # torch_shape = (4, 1536, 4096)
    # input = torch.empty(torch_shape, dtype=torch.float32, device=torch.cuda.current_device())

    input = data_provider(torch.cuda.current_device())[0]
    print_rank_0("input tensor shape: {}".format(input.shape))
    original_output = test_all_reduce_tensor_group(world_size, input.clone())

    """analyze quantized error between reduce scatter all gather and all reduce"""
    temp_output = test_reduce_scatter_all_gather_along_first_dim(world_size, input.clone())
    print_rank_0("analyze quantized error between reduce scatter all gather and all reduce")
    print_rank_0(analysis_diff(original_output, temp_output))

    """analyze quantized error between fake quantized all reduce and all reduce"""
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(original_output.contiguous())    
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    print_rank_0("analyze quantized error between fake quantized all reduce and all reduce")
    print_rank_0(analysis_diff(original_output, dequantized_tensor))

    """analyze quantized error between quantized all reduce """
    temp_output = test_quantized_all_reduce_tensor_group(world_size, input.clone())
    print_rank_0("analyze quantized error between quantized all reduce")
    print_rank_0(analysis_diff(original_output, temp_output))

def distributed_env_inatialize():
    world_size = Utils.world_size
    setup_timers()
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
if __name__ == '__main__':
    debug_test()