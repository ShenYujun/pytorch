# python3.7
"""Utility functions for optimizer and learning rate scheduler."""

from bisect import bisect_right

import torch.optim

__all__ = ['WarmupMultiStepLR', 'get_optimizer', 'get_lr_scheduler']


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
  """Defines a class for multi-step learning rate with warming up."""

  def __init__(self,
               optimizer,
               milestones,
               gamma=0.1,
               warmup_factor=0.0,
               warmup_steps=500,
               last_step=-1):
    """Initializes the class with specific settings.

    Note that different from official learning rate schedulers, this one takes
    `step` instead of `epoch` as the base unit.

    Args:
      optimizer: The optimizer for applying gradients.
      milestones: A list of steps with increasing order, indicating when to
        decay the learning rate.
      gamma: This field shows how much learning rate will be decayed each time.
      warmup_init: The initial warm-up factor.
      warmup_steps: Number of steps used for warming up.
      last_step: Previous training step.

    Raises:
      ValueError: If the milestones is not with increasing order.
    """
    milestones = list(milestones)
    if not milestones == sorted(milestones):
      raise ValueError(f'Milestones should be a list of increasing integers, '
                       f'but {milestones} received!')

    self.milestones = milestones
    self.gamma = float(gamma)
    self.warmup_factor = float(warmup_factor)
    self.warmup_steps = int(warmup_steps)
    super().__init__(optimizer, last_epoch=last_step)

  def get_lr(self):
    warmup_factor = 1.0
    if self.last_epoch < self.warmup_steps:
      alpha = self.last_epoch / self.warmup_steps
      warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
    lr_decay = warmup_factor * (
        self.gamma ** bisect_right(self.milestones, self.last_epoch))
    return [base_lr * lr_decay for base_lr in self.base_lrs]


def get_optimizer(config, model):
  """Gets optimizer."""
  params = []
  for key, val in model.named_parameters():
    if not val.requires_grad:
      continue
    lr = config.lr_base
    if 'bias' in key:
      lr = lr * config.lr_bias_factor
      weight_decay = config.weight_decay_bias
    else:
      weight_decay = config.weight_decay
    params.append({'params': [val], 'lr': lr, 'weight_decay': weight_decay})

  optimizer_type = config.optimizer_type.lower()
  if optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(params, lr, momentum=config.momentum)
  elif optimizer_type == 'adam':
    optimizer = torch.optim.Adam(params, lr)
  else:
    raise ValueError(f'Optimizer type `{optimizer_type}` is not supported!')

  return optimizer


def get_lr_scheduler(config, optimizer):
  """Gets learning rate scheduler."""
  return WarmupMultiStepLR(optimizer=optimizer,
                           milestones=config.lr_steps,
                           gamma=config.lr_decay,
                           warmup_factor=config.lr_warmup_factor,
                           warmup_steps=config.lr_warmup_steps)
