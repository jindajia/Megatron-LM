import torch
import megatron.core.parallel_state as ps
from megatron.timers import Timers
# from tests.unit_tests.test_utilities import Utils
import os 
from megatron.core.tensor_parallel.mappings import _reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim, _gather_along_last_dim, _reduce_scatter_along_last_dim

import sys
sys.path.append('/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_diff
from torch.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
import numpy as np
import fbgemm_gpu.quantize_comm
import struct

def data_provider(dev = 'cpu'):
    """import origin tensor"""
    tensor_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00300/layer_001/ParallelMLP/tensor_rank_0.pt'
    tensor = torch.load(tensor_path, map_location=torch.device(device=dev))
    return [tensor]


def analysis_data_info(input_tensor):
    num_nan_tensor = torch.numel(input_tensor[torch.isnan(input_tensor)])
    print("num NaN in tensor: {}, ratio: {}.".format(
            num_nan_tensor, num_nan_tensor / torch.numel(input_tensor)
        ))
    print("tensor profile: shape: {}, type: {}, sparsity: {}, min: {}, max: {}, min abs:{}, max abs:{}.".format(
        input_tensor.shape,
        input_tensor.dtype,
        calculate_sparsity(input_tensor),
        torch.min(input_tensor),
        torch.max(input_tensor),
        torch.min(torch.abs(input_tensor)),
        torch.max(torch.abs(input_tensor)),
    ))

def quantization_int8_per_token():
    print('-------quantization_int8_per_token-------')
    dev = torch.cuda.current_device()
    # dev = 'cpu'

    """load data"""
    input = data_provider(dev)[0] #shape: [1536, 4, 4096]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    row_dim = input.size(2) * input.size(1) # 4096 * 4
    original_shape = input.shape
    input_2d = input.view((-1, row_dim)) if row_dim > 0 else input # shape: [1536, 16384]
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d.contiguous())
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    dequantized_tensor = dequantized_tensor.view(original_shape)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

def quantization_int8_per_channel():
    print('-------quantization_int8_per_channel-------')
    dev = torch.cuda.current_device()
    # dev = 'cpu'

    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    input_trans = input.transpose(0, 2) # shape: [1536, 4, 4096] -> [4096, 4, 1536]
    row_dim = input.size(2) * input.size(1) # 1536 * 4
    original_shape = input_trans.shape
    input_2d = input_trans.reshape((-1, row_dim)) if row_dim > 0 else input_trans # this will drop performance
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d.contiguous())
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    dequantized_tensor = dequantized_tensor.view(original_shape)
    dequantized_tensor = dequantized_tensor.transpose(0, 2)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

def quantization_int8_per_tensor():
    print('-------quantization_int8_per_tensor-------')
    dev = torch.cuda.current_device()
    # dev = 'cpu'
    
    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    original_shape = input.shape
    input_2d = input.view(-1)
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d.contiguous())
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    last_eight = quantized_tensor[-8:].clone().cpu()
    scale_bits = last_eight[:4].numpy().tobytes()
    min_value_bits = last_eight[4:].numpy().tobytes()
    scale = struct.unpack('<f', scale_bits)[0]
    min_value = struct.unpack('<f', min_value_bits)[0]
    print(f"scale: {scale} min value: {min_value} (fbgemm will store min value, so that it can be used to calculate zero point)")

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    dequantized_tensor = dequantized_tensor.view(original_shape)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

def quantization_int8_default():
    print('-------quantization_int8_default-------')
    dev = torch.cuda.current_device()
    # dev = 'cpu'
    
    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    original_shape = input.shape
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input.contiguous())
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))
    dequantized_tensor = dequantized_tensor.view(original_shape)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

"""e4m3 when is_fwd is True, e5m2 when is_fwd is False"""
def quantization_fp8_per_tensor(is_fwd=True):
    print('-------quantization_fp8_per_tensor is_fwd{}-------'.format(str(is_fwd)))
    dev = torch.cuda.current_device()
    # dev = 'cpu'
    
    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    original_shape = input.shape
    input_2d = input.view((-1))
    quantized_tensor = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
        input_2d.contiguous(), is_fwd
    )
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.FP8RowwiseQuantizedToFloat(
        quantized_tensor, is_fwd
    )
    dequantized_tensor = dequantized_tensor.view(original_shape)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

"""e4m3 when is_fwd is True, e5m2 when is_fwd is False"""
def quantization_fp8_per_token(is_fwd=True):
    print('-------quantization_fp8_per_token is_fwd{}-------'.format(str(is_fwd)))
    dev = torch.cuda.current_device()
    # dev = 'cpu'
    
    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    row_dim = 4096
    original_shape = input.shape
    input_2d = input.view((-1, row_dim)) if row_dim > 0 else input
    quantized_tensor = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
        input_2d.contiguous(), is_fwd
    )
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.FP8RowwiseQuantizedToFloat(
        quantized_tensor, is_fwd
    )
    dequantized_tensor = dequantized_tensor.view(original_shape)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

"""e4m3 when is_fwd is True, e5m2 when is_fwd is False"""
def quantization_fp8_per_channel(is_fwd=True):
    print('-------quantization_fp8_per_channel is_fwd{}-------'.format(str(is_fwd)))
    dev = torch.cuda.current_device()
    # dev = 'cpu'
    
    """load data"""
    input = data_provider(dev)[0]
    print("input tensor shape: {}, device: {}".format(input.shape, input.device))

    """quantize data"""
    row_dim = 1536
    input_trans = input.transpose(0, 2)
    original_shape = input_trans.shape
    input_2d = input_trans.reshape((-1, row_dim)) if row_dim > 0 else input_trans # this will drop performance
    quantized_tensor = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
        input_2d.contiguous(), is_fwd
    )
    print("quantize tensor shape: {}, type: {}".format(quantized_tensor.shape, quantized_tensor.dtype))

    """dequantize data"""
    dequantized_tensor = torch.ops.fbgemm.FP8RowwiseQuantizedToFloat(
        quantized_tensor, is_fwd
    )
    dequantized_tensor = dequantized_tensor.view(original_shape)
    dequantized_tensor = dequantized_tensor.transpose(0, 2)
    print("dequantized tensor shape: {}, type: {}".format(dequantized_tensor.shape, dequantized_tensor.dtype))

    """error calculation"""
    print(analysis_diff(input, dequantized_tensor))

def baseline_torch_per_tensor_quantization():
    tensor = data_provider('cpu')[0].to(torch.float32)

    """MinMaxObserver"""
    print('baseline_min_max_observer')
    qscheme = torch.per_tensor_affine
    obs = MinMaxObserver(qscheme=qscheme, dtype=torch.quint8)
    obs(tensor)
    scale, zero_point = obs.calculate_qparams()
    print(f"Qscheme: {qscheme} | scale: {scale} zero_point: {zero_point}")
    quantized_tensor = torch.quantize_per_tensor(tensor, scale.item(), zero_point.item(), torch.quint8)
    print(analysis_diff(tensor, quantized_tensor.dequantize()))

    """PerChannelMinMaxObserver axis = 0"""
    print('PerChannelMinMaxObserver axis = 0')
    channel_index = 0
    qscheme = torch.per_channel_affine
    obs = PerChannelMinMaxObserver(qscheme=qscheme, dtype=torch.quint8, ch_axis=channel_index)
    obs(tensor)
    scale, zero_point = obs.calculate_qparams()
    print(f"Qscheme: {qscheme} | scale shape: {scale.shape} zero_point shape: {zero_point.shape}")
    quantized_tensor = torch.quantize_per_channel(tensor, scale, zero_point, channel_index, torch.quint8)
    print(analysis_diff(tensor, quantized_tensor.dequantize()))

    """PerChannelMinMaxObserver axis = 2"""
    print('PerChannelMinMaxObserver axis = 2')
    channel_index = 2
    qscheme = torch.per_channel_affine
    obs = PerChannelMinMaxObserver(qscheme=qscheme, dtype=torch.quint8, ch_axis=channel_index)
    obs(tensor)
    scale, zero_point = obs.calculate_qparams()
    print(f"Qscheme: {qscheme} | scale shape: {scale.shape} zero_point shape: {zero_point.shape}")
    quantized_tensor = torch.quantize_per_channel(tensor, scale, zero_point, channel_index, torch.quint8)
    print(analysis_diff(tensor, quantized_tensor.dequantize()))

if __name__ == '__main__':
    print('baseline tensor info')
    input = data_provider('cpu')[0]
    analysis_data_info(input)

    baseline_torch_per_tensor_quantization()
    # quantization_int8_default()
    quantization_int8_per_tensor()
    quantization_int8_per_token()
    quantization_int8_per_channel()
    # quantization_fp8_per_tensor()
    # quantization_fp8_per_token()
    # quantization_fp8_per_channel()



