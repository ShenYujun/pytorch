# python3.7
"""Utility functions for distributed running."""

import pickle

import torch
import torch.distributed as dist
from torch.utils.collect_env import get_pretty_env_info

from configs import CPU_DEVICE, GPU_DEVICE

__all__ = ['get_rank', 'get_world_size', 'get_env_info', 'reduce', 'gather',
           'synchronize']


def get_rank():
  """Gets rank of current process."""
  if not dist.is_available() or not dist.is_initialized():
    return 0
  return dist.get_rank()


def get_world_size():
  """Gets world size (total number of GPUs used)."""
  if not dist.is_available() or not dist.is_initialized():
    return 1
  return dist.get_world_size()


def get_env_info():
  """Gets the environment information."""
  return '\n' + get_pretty_env_info()


def reduce(tensor_dict, average=True):
  """Reduces a dictionary of tensors from all processes to that with rank 0."""
  world_size = get_world_size()
  if world_size == 1:
    return tensor_dict

  with torch.no_grad():
    names = []
    tensors = []
    for key in sorted(tensor_dict.keys()):
      names.append(key)
      tensors.append(tensor_dict[key])
    tensors = torch.stack(tensors, dim=0)
    dist.reduce(tensors, dst=0)
    if get_rank() == 0 and average:
      tensors = tensors / world_size
    reduced_dict = {name: tensor for name, tensor in zip(names, tensors)}

  return reduced_dict


def gather(data_dict):
  """Gathers a dictionary of data from all processes.

  There are some things need to be clarified:
  1. Data does not need to be `torch.Tensor`. Specifically, this function draws
     support from package `pickle`. So, please make sure the data can be
     pickled.
  2. This function is commonly used for testing, where the results from
     different GPUs are gathered together. Accordingly, please use unique key
     for each sample.
  3. This function will return a whole dictionary on CPU, instead of passing it
     to any rank of process.

  Args:
    data_dict: A dictionary of data to gather. Key is recommended to be able to
      uniquely identify the sample (or the process).

  Returns:
    A gathered dictionary with data from all processes.
  """
  world_size = get_world_size()
  if world_size == 1:
    return data_dict

  # Serialize data to `torch.ByteTensor`.
  storage = torch.ByteStorage.from_buffer(pickle.dumps(data_dict))
  tensor = torch.ByteTensor(storage).to(GPU_DEVICE)

  # Obtain maximum size and do padding to align all data.
  local_size = torch.IntTensor([tensor.numel()]).to(GPU_DEVICE)
  size_list = [torch.IntTensor([0]).to(GPU_DEVICE) for _ in range(world_size)]
  dist.all_gather(size_list, local_size)
  size_list = [int(size.item()) for size in size_list]
  max_size = max(size_list)
  if local_size < max_size:
    padding = torch.ByteTensor(size=(max_size - local_size,)).to(GPU_DEVICE)
    tensor = torch.cat((tensor, padding), dim=0)
  tensor_list = [torch.ByteTensor(size=(max_size,)).to(GPU_DEVICE)
                 for _ in range(world_size)]
  dist.all_gather(tensor_list, tensor)

  # Deserialize `torch.ByteTensor` to data.
  gathered_dict = {}
  for size, tensor in zip(size_list, tensor_list):
    gathered_dict.update(
        pickle.loads(tensor.to(CPU_DEVICE).numpy().tobytes()[:size]))

  return gathered_dict


def synchronize():
  """Synchronizes all processes."""
  if get_world_size() == 1:
    return
  dist.barrier()
