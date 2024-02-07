import torch
import megatron.core.parallel_state as ps
from megatron.timers import Timers
from tests.unit_tests.test_utilities import Utils
import os 
from megatron.core.tensor_parallel.mappings import _reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim, _gather_along_last_dim, _reduce_scatter_along_last_dim
import torch.multiprocessing as mp
import fbgemm_gpu
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_world_size
)

def print_all(msg):
    msgs = [None for _ in range(torch.cuda.device_count())]
    msg = ('rank {}'.format(torch.distributed.get_rank()), msg)
    torch.distributed.all_gather_object(msgs,
                                        msg)
    if torch.distributed.get_rank() == 0:
        for msg in msgs:
            print(msg)

def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def test_all_gather():
    if torch.distributed.get_rank() == 0:
        torch_shape = (4, 1)
    else:
        torch_shape = (2, 1)
    input = torch.empty(torch_shape, dtype=torch.half, device=torch.cuda.current_device())
    torch.rand(out=input, dtype=torch.half, size=torch_shape)
    print_all(input)

    """should all gather tensor shape first"""
    torch_shape = input.shape
    tensor_shape_list = [None for _ in range(get_tensor_model_parallel_world_size())]
    torch.distributed.all_gather_object(tensor_shape_list, torch_shape, group=get_tensor_model_parallel_group())

    """create all gather output buffer with tensor_shape_list"""
    tensor_list = [torch.empty(buffer_shape, dtype=torch.half, device=torch.cuda.current_device()) for buffer_shape in tensor_shape_list]
    torch.distributed.all_gather(tensor_list, input, group=get_tensor_model_parallel_group())
    
    print_rank_0(tensor_list)


if __name__ == '__main__':
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    test_all_gather()
