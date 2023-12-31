from tools.deepspeed.ops.op_builder.quantizer import CUDAQuantizer
import torch

class QuantizeHelper:
    def __init__(self, quantize_weights=True, quantize_gradients=True, bucket_size=2048):
        """
        Initialize the Quantization Helper.

        Args:
            quantize_weights (bool): If True, quantize weights.
            quantize_gradients (bool): If True, quantize gradients.
        """
        self.quantize_weights = quantize_weights
        self.quantize_gradients = quantize_gradients
        self.cuda_quantizer = CUDAQuantizer()
        self.cuda_quantizer.target_group_size = bucket_size
        self.quantized_buffer = None

    def allocate_buffer(self, expected_shape, dtype=torch.int8):
        """
        Allocate the quantized buffer.

        Args:
            expected_shape (torch.Size or int): The expected shape or size of the buffer for all-gather.
            dtype (torch.dtype): Data type of the quantized buffer.
        """
        if isinstance(expected_shape, int):
            expected_size = expected_shape
        elif isinstance(expected_shape, torch.Size):
            expected_size = torch.Size(expected_shape).numel()
        else:
            raise ValueError("Expected shape must be a torch.Size or int")
        if self.quantized_buffer is None or self.quantized_buffer.numel() < expected_size:
            print("JINDA_DEBUG allocate quantize buffer")
            self.quantized_buffer = torch.empty(expected_size, dtype=dtype, device=torch.cuda.current_device())


    def quantize(self, tensor):
        """
        Quantize the given tensor using CUDAQuantizer.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.

        Returns:
            quantized_param: The quantized tensor.
            scales: quantized scales
        """
        quantized_param, scales = self.cuda_quantizer.quantize(tensor)
        return quantized_param, scales

    def dequantize(self, quantized_tensor, scale, received_buffer=None):
        """
        Dequantize the given tensor using CUDAQuantizer.

        Args:
            quantized_tensor (torch.Tensor): The tensor to be dequantized.
            scale (float): Scale factor for dequantization.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        if received_buffer is not None:
            received_buffer.data = self.cuda_quantizer.dequantize(quantized_tensor, scale)
            return received_buffer
        else:
            return self.cuda_quantizer.dequantize(quantized_tensor, scale)
