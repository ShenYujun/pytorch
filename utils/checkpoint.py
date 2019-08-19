# python3.7
"""Contains the class to help handle checkpoint-related matters."""

import os.path
import torch

__all__ = ['Checkpointer']


def strip_prefix(state_dict, prefix='module.'):
  """Removes the prefix for model deployed by `DistributedDataParallel`."""
  if not all(key.startswith(prefix) for key in state_dict.keys()):
    return state_dict
  stripped_state_dict = {}
  for key in list(state_dict.keys()):
    stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
  return stripped_state_dict


class Checkpointer(object):
  """Class to handle checkpoint, including both saving and loading."""

  def __init__(self, work_dir, model, optimizer, lr_scheduler, logger):
    """Initializes the class.

    Args:
      work_dir: Work directory to save the checkpoint.
      model: The model with parameters to save or load.
      optimizer: The optimizer, for training only.
      lr_scheduler: The learning rate scheduler, for training only.
      logger: The logger to log message.
    """
    self.work_dir = work_dir
    self.model = model
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.logger = logger
    self.last_tagfile = 'last_checkpoint'

  def tag_last_checkpoint(self, last_checkpoint):
    """Tags the last saved checkpoint."""
    with open(os.path.join(self.work_dir, self.last_tagfile), 'w') as f:
      f.write(last_checkpoint)

  def has_checkpoint(self, dirname):
    """Checks whether a directory contains checkpoint or not."""
    return os.path.exists(os.path.join(dirname, self.last_tagfile))

  def get_last_checkpoint(self, dirname):
    """Gets the last checkpoint file from given directory."""
    with open(os.path.join(dirname, self.last_tagfile), 'r') as f:
      last_checkpoint = f.read().strip()
    return os.path.join(dirname, last_checkpoint)

  def save(self, filename, step, **kwargs):
    """Saves model state to target file."""
    data = {}
    data['model'] = self.model.state_dict()
    if self.optimizer:
      data['optimizer'] = self.optimizer.state_dict()
    if self.lr_scheduler:
      data['lr_scheduler'] = self.lr_scheduler.state_dict()
    data['step'] = step
    data.update(kwargs)

    filename = os.path.join(self.work_dir, filename)
    torch.save(data, filename)
    self.tag_last_checkpoint(filename)
    self.logger.info(f'Successfully saved checkpoint to {filename}!')

  def load(self, file_or_dir, weights_only=False):
    """Loads model state from file or directory."""
    if not os.path.exists(file_or_dir):
      self.logger.warning(f'`{file_or_dir}` does not exist! Skip loading.')
      return {}
    if os.path.isfile(file_or_dir):
      load_path = file_or_dir
    if os.path.isdir(file_or_dir):
      if not self.has_checkpoint(file_or_dir):
        self.logger.warning(f'No checkpoint found in `{file_or_dir}`!'
                            f'Skip loading.')
        return {}
      load_path = self.get_last_checkpoint(file_or_dir)

    data = torch.load(load_path, map_location=torch.device('cpu'))
    if 'model' not in data:
      data = {'model': data}
    model_state_dict = self.model.state_dict()
    loaded_state_dict = strip_prefix(data.pop('model'))
    for key, val in loaded_state_dict.items():
      self.logger.debug(f'Loading {key} with shape {tuple(val.shape)}.')
    model_state_dict.update(loaded_state_dict)
    self.model.load_state_dict(model_state_dict)
    if self.optimizer and 'optimizer' in data and not weights_only:
      self.optimizer.load_state_dict(data.pop('optimizer'))
      self.logger.info(f'Successfully loaded optimizer from `{load_path}`!')
    if self.lr_scheduler and 'lr_scheduler' in data and not weights_only:
      self.lr_scheduler.load_state_dict(data.pop('lr_scheduler'))
      self.logger.info(f'Successfully loaded lr scheduler from `{load_path}`!')
    self.logger.info(f'Successfully loaded checkpoint from `{load_path}`!')

    return data if not weights_only else {}
