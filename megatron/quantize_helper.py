from tools.deepspeed.ops.op_builder.quantizer import CUDAQuantizer
import torch

class QuantizeHelper:
    def __init__(self, quantize_weights=True, weight_quantization_bits = 4, wq_bucket_size=2048, quantize_gradients=True, gradeint_quantization_bits=8):
        """
        Initialize the Quantization Helper.

        Args:
            quantize_weights (bool): If True, quantize weights.
            quantize_gradients (bool): If True, quantize gradients.
        """
        self.quantize_weights = quantize_weights
        self.weight_quantization_bits = weight_quantization_bits
        self.wq_bucket_size = wq_bucket_size
        self.quantize_gradients = quantize_gradients
        self.gradeint_quantization_bits = gradeint_quantization_bits
        self.cuda_quantizer = CUDAQuantizer()
        self.cuda_quantizer.target_group_size = wq_bucket_size

    def quantize_gather_weights(self, weight_tensor):
        """
        Quantize the given tensor using CUDAQuantizer.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.

        Returns:
            quantized_param: The quantized tensor.
            scales: quantized scales
        """
        quantized_param, scales = self.cuda_quantizer.quantize(weight_tensor, quantization_bits=self.weight_quantization_bits)
        return quantized_param, scales

    def dequantize_gather_weights(self, quantized_weight_tensor, scale, received_buffer=None):
        """
        Dequantize the given tensor using CUDAQuantizer.

        Args:
            quantized_tensor (torch.Tensor): The tensor to be dequantized.
            scale (float): Scale factor for dequantization.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        if received_buffer is not None:
            received_buffer.data = self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits)
            return received_buffer
        else:
            return self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits)
