import torch

def reshape_to_2d(dim):
    prod_of_dims = int(torch.prod(torch.tensor(dim[:-1], dtype=torch.int)).item())
    new_dim = (prod_of_dims, dim[-1])
    return new_dim