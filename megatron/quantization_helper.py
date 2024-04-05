from tools.deepspeed.ops.op_builder.quantizer import CUDAQuantizer, QuantizerBuilder
import torch
from torch.distributed import ProcessGroup, all_to_all_single
import math
import os
from .quantization_cuda_builder import find_module, build_module

def build_or_import_siwzzle_quant_module():
    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../','tools/jet_quant_cuda')
    module_name = 'quantization_cuda'
    module = find_module(pkg_path, module_name)
    if module is not None:
        return module
    else:
        if torch.distributed.is_initialized() and torch.distributed.get_rank()==0:
            build_module(pkg_path)
        torch.distributed.barrier()
        module = find_module(pkg_path, module_name)
        return module

def analysis_diff(origin_tensor, quantized_tensor):

    diff = origin_tensor - quantized_tensor
    abs_error_norm = torch.norm(diff)
    origin_norm = torch.norm(origin_tensor)
    rela_error_norm = abs_error_norm / origin_norm
    return abs_error_norm, rela_error_norm

def get_hadamard_matrix(k):
    T = {}
    H = torch.ones((1, 1), device=torch.cuda.current_device(), dtype=torch.float)
    T[0] = H
    for i in range(1, k+1):
        H = torch.cat((torch.cat([H, H], 1),
                    torch.cat([H, -H], 1)), 0)
        T[i] = H
    return T[k]

class QuantizationHelper:
    def __init__(self, quantized_weights=True, 
                 weight_quantization_bits = 4, 
                 wq_group_size=2048, 
                 quantized_gradients=True, 
                 gradient_quantization_bits_inter=4,
                 gq_group_size_inter=128,
                 gradient_quantization_bits_intra=8,
                 gq_group_size_intra=512,
                 data_parallel_group: torch.distributed.ProcessGroup = None,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 hadamard_transform=False,
                 ):

        self.quantized_weights = quantized_weights
        self.weight_quantization_bits = weight_quantization_bits
        self.wq_group_size = wq_group_size
        self.quantized_gradients = quantized_gradients
        self.gradient_quantization_bits_inter = gradient_quantization_bits_inter
        self.gq_group_size_inter = gq_group_size_inter
        self.gradient_quantization_bits_intra = gradient_quantization_bits_intra
        self.gq_group_size_intra = gq_group_size_intra
        self.data_parallel_group = data_parallel_group
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.hadamard_transform = hadamard_transform
        self.hadamard_order = 5
        self.hadamard_group_size = 2**self.hadamard_order
        self.hadamard_matrix = None
        if self.quantized_gradients or self.quantized_weights:
            self.set_local_all_to_all_group()
            self.quant_module = self.build_or_import_siwzzle_quant_module()

    def build_or_import_siwzzle_quant_module(self):
        pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../','tools/jet_quant_cuda')
        module_name = 'quantization_cuda'
        module = find_module(pkg_path, module_name)
        if module is not None:
            return module
        else:
            if torch.distributed.is_initialized() and torch.distributed.get_rank()==0:
                build_module(pkg_path)
            torch.distributed.barrier()
            module = find_module(pkg_path, module_name)
            return module
     
    def set_local_all_to_all_group(self):
        assert torch.distributed.is_initialized(), 'dist is not initialized'
        all_to_all_group = {}
        data_parallel_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        tensor_model_parallel_size = self.tensor_parallel_size
        pipeline_model_parallel_size =self.pipeline_parallel_size
        gpus_per_node = int(os.environ['LOCAL_WORLD_SIZE'])
        local_dp_size = gpus_per_node // (tensor_model_parallel_size * pipeline_model_parallel_size) # data parallel size in a node
        if local_dp_size < 1:
            local_dp_size = 1
        self.local_world_size = local_dp_size
        num_local = data_parallel_world_size // local_dp_size
        self.num_nodes = num_local

        for i in range(num_local):
            local_rank = [j + local_dp_size * i for j in range(local_dp_size)]
            all_to_all_group[f"local_{i}"] = torch.distributed.new_group(ranks=local_rank)

        for i in range(local_dp_size):
            cur_rank = []
            for j in range(num_local):
                cur_rank.append(i + j * local_dp_size)
            all_to_all_group[f"global_{i}"] = torch.distributed.new_group(ranks=cur_rank)
        self.all2all_process_group = all_to_all_group

    def quantize_gather_weights(self, weight_tensor :torch.Tensor):
        """
        Quantize the given tensor using CUDAQuantizer.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.

        Returns:
            quantized_param: The quantized tensor.
            scales: quantized scales
        """

        assert weight_tensor.nelement() % self.wq_group_size == 0
        groups =  weight_tensor.nelement() // self.wq_group_size
        quant_module = self.quant_module
        if weight_tensor.dtype is not torch.half or weight_tensor.dtype is not torch.float:
            weight_tensor = weight_tensor.to(torch.float)
        quant_tensor, quant_scales = quant_module.stochastic_quantize(weight_tensor, groups, self.weight_quantization_bits, quant_module.Symmetric)
        return quant_tensor, quant_scales

    def dequantize_gather_weights(self, quantized_weight_tensor, scales, dequant_type, received_buffer=None):
        """
        Dequantize the given tensor using CUDAQuantizer.

        Args:
            quantized_tensor (torch.Tensor): The tensor to be dequantized.
            scale (float): Scale factor for dequantization.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        if self.weight_quantization_bits == 4:
            quantized_weight_tensor = self.use_2int4_represent_1int8(quantized_weight_tensor)
        dequant_value = self.dequantize_nbits(quantized_weight_tensor, scales, groupsize=self.wq_group_size)
        if dequant_value.dtype is not dequant_type:
            dequant_value = dequant_value.to(dequant_type)
        if received_buffer is not None:
            received_buffer.copy_(dequant_value)
            return received_buffer
        else:
            return dequant_value

    def quantize_reduce_gradients(self, tensor, received_buffer=None):
        world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        # when grad type float16, should use fp32 to do transformation and quantization
        original_grad_type = tensor.dtype
        if original_grad_type is not torch.float32:
            tensor = tensor.to(torch.float32)
        if self.hadamard_transform:
            tensor = self.hadamard_tranformation_grad(tensor)
        final_output = self.quantized_reduce_scatter(tensor)
        if self.hadamard_transform:
            final_output = self.hadamard_back_tranformation(final_output)
        if final_output.dtype is not original_grad_type:
            final_output = final_output.to(original_grad_type)
        received_buffer.copy_(final_output)
    
    def _all_to_all_along_first_dim(self, input_, output=None):
        """All to All gather tensor"""
        world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        if world_size == 1:
            return input_
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by data parallel size"

        # Prepare output tensor
        if output is None:
            dim_size = list(input_.size())
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        
        # Perform the all_to_all_single operation
        torch.distributed.all_to_all_single(output, input_, group=self.data_parallel_group)
        return output

    def set_gradient_quantization(self, quantize_gradients: bool):
        self.quantized_gradients = quantize_gradients

    def hadamard_tranformation_grad(self, tensor:torch.Tensor):
        group_size = self.hadamard_group_size
        # split tensor to groups
        tensor = tensor.view(-1, group_size)

        # create Hadamard matrix
        if self.hadamard_matrix is None:
            self.hadamard_matrix = get_hadamard_matrix(self.hadamard_order)
        H = self.hadamard_matrix

        transformed_tensor = (tensor @ H) / torch.tensor(group_size, device=torch.cuda.current_device())

        return transformed_tensor.view(-1)

    def hadamard_back_tranformation(self, transformed_tensor):
        group_size = self.hadamard_group_size

        # split tensor to groups
        transformed_tensor = transformed_tensor.view(-1, group_size)

        H = self.hadamard_matrix

        original_tensor = (transformed_tensor @ H)

        return original_tensor.view(-1)

    def quantized_reduce_scatter(self, tensor):
        assert tensor.numel() % self.gq_group_size_inter == 0 # tensor size must be multiple of group size
        assert self.gq_group_size_inter  % (8 // min(self.gradient_quantization_bits_inter, self.gradient_quantization_bits_intra)) == 0 # group size must be multiple of 2 when using 4bits
        assert self.gq_group_size_inter % 8 == 0 # group size must be multiple of 8 when tensor is half type; must be multiple of 4 when type is float. 
                                    # That is because cuda swizzle quant function will load 4 float or 8 half for each thread step to get better performance
        # assert (tensor.numel() // self.gq_group_size_inter) % (num_nodes * local_world_size * pipeline) == 0
        groups = self.all2all_process_group
        global_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        # group_size = self.gq_group_size
        intra_quant_group = max(math.ceil(tensor.numel() / self.gq_group_size_intra), global_world_size)
        local_world_size = self.local_world_size
        num_nodes = self.num_nodes
        inter_quant_group = intra_quant_group // local_world_size
        this_rank = torch.distributed.get_rank(
            group=self.data_parallel_group
        )
        intra_idx = int(this_rank / local_world_size)
        inter_idx = this_rank % local_world_size
        quant_module = self.quant_module

        """intra node quantization and all-to-all"""
        output_tensor, output_scales = quant_module.swizzle_quant(  tensor, 
                                                                            intra_quant_group, 
                                                                            self.gradient_quantization_bits_intra,
                                                                            quant_module.Symmetric, 
                                                                            1, 
                                                                            num_nodes,
                                                                            local_world_size)
        """all to all, dequantReduction"""
        all_to_all_output_tensor = torch.empty_like(output_tensor)
        all_to_all_output_scales = torch.empty_like(output_scales)
        all_to_all_single(all_to_all_output_tensor, output_tensor, group=groups[f'local_{intra_idx}'])
        all_to_all_single(all_to_all_output_scales, output_scales, group=groups[f'local_{intra_idx}'])

        reduced_tensor, = quant_module.quantized_reduction(
            all_to_all_output_tensor, all_to_all_output_scales, intra_quant_group, inter_quant_group, self.gradient_quantization_bits_intra, quant_module.Symmetric,
            local_world_size)
        
        """inter node quantization and all-to-all"""
        quant_tensor, quant_scales = quant_module.stochastic_quantize(reduced_tensor, inter_quant_group, self.gradient_quantization_bits_inter, quant_module.Symmetric)

        """all to all"""
        all_to_all_output_tensor = torch.empty_like(quant_tensor)
        all_to_all_output_scales = torch.empty_like(quant_scales)
        all_to_all_single(all_to_all_output_tensor, quant_tensor, group=groups[f'global_{inter_idx}'])
        all_to_all_single(all_to_all_output_scales, quant_scales, group=groups[f'global_{inter_idx}'])

        """dequantizeReduction"""
        dquant_output, = quant_module.quantized_reduction(all_to_all_output_tensor, 
                                                            all_to_all_output_scales, 
                                                            inter_quant_group, 
                                                            inter_quant_group // num_nodes, 
                                                            self.gradient_quantization_bits_inter, 
                                                            quant_module.Symmetric,
                                                            num_nodes)

        return dquant_output

    def quantize_4bits(self, x, groupsize=-1):
        bits = 4

        assert len(list(x.shape)) == 1
        assert groupsize % 2 == 0
        x_shape = list(x.size())[0]
        d = 2 ** (bits - 1)-1 ###

        if groupsize == -1:
            norm = torch.max(torch.abs(x))
            group_x = x
        else:
            assert list(x.shape)[0] % groupsize == 0
            group_x = x.view(
                -1,
                groupsize,
            )
            norm, _ = torch.max(group_x.abs(), -1, keepdim=True)
            norm[norm==0] = 2 ** (bits - 1) - 1 ###

        # level_float = d * torch.abs(group_x) / norm
        level_float = d * torch.clamp(torch.abs(group_x) / norm, max=1)
        previous_level = torch.floor(level_float)
        # is_next_level = torch.rand(group_x.size()).to(group_x.device) < (level_float - previous_level)
        is_next_level = torch.rand(group_x.size(), device=group_x.device) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        scale = norm / d
        scale = scale.view(torch.int8)
        x_quant = torch.sign(group_x) * new_level
        x_quant = x_quant.to(torch.int8)
        x_quant = x_quant.view(x_shape)
        # print('x_quant before tensor:', x_quant)
        x_quant = self.use_1int8_represent_2int4(int4_input=x_quant).view(-1, groupsize // 2)

        # print('x_scale before tensor:', scale.view(torch.float32))

        # return torch.cat((x_quant, scale), 1)
        return x_quant, scale

    def dequantize_4bits(self, x, s, groupsize=-1):

        x = self.use_2int4_represent_1int8(x).to(torch.float32)
        s = s.view(torch.float32)
        # print('x_scale tensor:', s.view(torch.float32))
        # print('x_quant', x)

        if groupsize == -1:
            group_x = x
        else:
            group_x = x.view(
                -1,
                groupsize,
            )
        group_x.mul_(s)
        x_dequant = group_x.view(-1)

        return x_dequant

    def dequantize_nbits(self, x: torch.int8, s:torch.float32, groupsize=-1):
        x = x.to(torch.float32)
        s = s.view(torch.float32).view(-1, 1)

        if groupsize == -1:
            group_x = x
        else:
            group_x = x.view(
                -1,
                groupsize,
            )
        group_x.mul_(s)
        x_dequant = group_x.view(-1)

        return x_dequant

    def use_1int8_represent_2int4(self, int4_input):
        assert len(list(int4_input.shape)) == 1
        assert list(int4_input.shape)[0] % 2 == 0
        half = list(int4_input.shape)[0] // 2
        a, b = int4_input[::2], int4_input[1::2]

        packed = (a << 4) | (b & 0b00001111)

        return packed

    def use_2int4_represent_1int8(self, int8_input):
        a_unpacked = int8_input >> 4
        b_unpacked = int8_input << 4 >> 4

        unpacked = torch.stack((a_unpacked.view(-1), b_unpacked.view(-1))).transpose(0, 1).flatten()

        return unpacked