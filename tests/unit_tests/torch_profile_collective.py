import torch
import torch.distributed as dist
from tests.unit_tests.test_utilities import Utils
from megatron.timers import Timers
from torch.profiler import profile, record_function, ProfilerActivity

import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

_GLOBAL_TIMERS = None

def setup_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    assert _GLOBAL_TIMERS is None, '{} is already initialized.'.format('timers')
    _GLOBAL_TIMERS = Timers(2, 'all')

def get_timers():
    """Return timers."""
    assert _GLOBAL_TIMERS is not None, '{} is not initialized.'.format('timers')
    return _GLOBAL_TIMERS

def distributed_env_inatialize():
    world_size = Utils.world_size
    setup_timers()
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)

def profile_all_reduce():
    """create random tensor"""
    torch_shape = (4, 1536, 4096)
    input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
    copy_input = input.clone()

    """warm up"""
    for i in range(5):
        dist.all_reduce(input)
        input.copy_(copy_input)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dist.all_reduce(input)
    
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

if __name__ == '__main__':
    distributed_env_inatialize()
    profile_all_reduce()