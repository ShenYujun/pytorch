# python3.7
"""Defines data samplers."""

from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.distributed import DistributedSampler


__all__ = ['get_batch_sampler']


class FixedStepBatchSampler(Sampler):
  """Warps batch sampler with fixed number of steps."""

  def __init__(self, batch_sampler, max_step, start_step=0):
    assert max_step > 0
    self.batch_sampler = batch_sampler
    self.max_step = max_step
    self.start_step = start_step

  def __iter__(self):
    step = self.start_step
    while step <= self.max_step:
      if hasattr(self.batch_sampler.sampler, 'set_epoch'):
        self.batch_sampler.sampler.set_epoch(step)
      for batch in self.batch_sampler:
        step += 1
        if step > self.max_step:
          break
        yield batch

  def __len__(self):
    return self.max_step


def get_batch_sampler(dataset, shuffle, max_step, start_step, config):
  """Gets data sampler.

  Args:
    dataset: A `torch.utils.data.Dataset` object.
    shuffle: Whether to shuffle the dataset.
    max_step: Maximum number of running steps. For training only.
    start_step: Step at which to start running. For training only.
    config: Other configurations.

  Returns:
    A `torch.utils.data.sampler.Sampler` object.
  """
  if config.is_distributed:
    sampler = DistributedSampler(dataset)
  elif shuffle:
    sampler = RandomSampler(dataset)
  else:
    sampler = SequentialSampler(dataset)

  batch_sampler = BatchSampler(
      sampler, config.batch_size_per_gpu, drop_last=False)
  if max_step != 0:
    batch_sampler = FixedStepBatchSampler(batch_sampler, max_step, start_step)

  return batch_sampler
