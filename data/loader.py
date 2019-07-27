# python3.7
"""Utility functions for datasets and data transforms."""

from torch.utils.data import DataLoader

from data.datasets import get_dataset
from data.transforms import get_transform
from data.samplers import get_batch_sampler

__all__ = ['get_data_loader']


def get_data_loader(config, start_step=0):
  """Gets data loader for data prefetching."""
  if config.run_mode == 'train':
    shuffle = True
    max_step = config.max_step
    dataset_name = config.train_dataset_name
    image_dir = config.train_image_dir
    label_file = config.train_label_file
  elif config.run_mode == 'test':
    shuffle = config.is_distributed
    max_step = 0
    start_step = 0
    dataset_name = config.test_dataset_name
    image_dir = config.test_image_dir
    label_file = config.test_label_file

  transforms = get_transform(transform_name=config.data_transform,
                             config=config)
  dataset = get_dataset(dataset_name=dataset_name,
                        image_dir=image_dir,
                        label_file=label_file,
                        transforms=transforms)
  batch_sampler = get_batch_sampler(dataset=dataset,
                                    shuffle=shuffle,
                                    max_step=max_step,
                                    start_step=start_step,
                                    config=config)
  data_loader = DataLoader(dataset=dataset,
                           num_workers=config.num_workers_per_gpu,
                           batch_sampler=batch_sampler)

  return data_loader
