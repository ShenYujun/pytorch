# python3.7
"""Contains losses and accuracies."""

import torch
import torch.nn as nn

__all__ = ['get_loss']


LOSSES = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss,
    'softmax': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss,  # negative log likelihood loss.
    'poisson_nll': nn.PoissonNLLLoss,
    'kl': nn.KLDivLoss,
}


def get_loss(loss_type, **kwargs):
  """Gets loss by type."""
  loss_type = loss_type.lower()
  try:
    loss_fn = LOSSES[loss_type](**kwargs)
  except KeyError:
    raise ValueError(f'Loss type `{loss_type}` is not supported!\n'
                     f'Please choose from {list(LOSSES)}.')
  return loss_fn


def accuracy(outputs, targets, top_k=(1,)):
  """Computes the accuracy over the top-k predictions."""
  if isinstance(top_k, int):
    top_k = (top_k,)
  max_k = max(top_k)

  with torch.no_grad():
    batch_size = outputs.shape[0]
    assert batch_size == targets.shape[0]

    _, predictions = outputs.topk(max_k, dim=1)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    results = []
    for k in top_k:
      correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
      results.append(correct_k.mul_(100 / batch_size))

    return results
