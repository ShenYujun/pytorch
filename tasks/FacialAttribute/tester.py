# python3.7
"""Contains testing tools."""

import os.path
from time import time
from datetime import timedelta

import torch

# pylint: disable=import-error, no-name-in-module
from configs import CPU_DEVICE, GPU_DEVICE
from models.model_handler import get_model
from data.loader import get_data_loader
from tools.losses import accuracy
import utils.checkpoint as ckpt_utils
import utils.distribute as dist_utils
# pylint: enable=import-error, no-name-in-module

__all__ = ['test']


def test(config, logger, model=None):
  """Model testing."""
  is_chef = (dist_utils.get_rank() == 0)

  if model is None:
    logger.info(f'Deploy model.')
    model = get_model(model_name=config.model_structure,
                      pretrained=config.use_pretrain,
                      num_classes=config.num_classes)
    model.to(GPU_DEVICE)

    checkpointer = ckpt_utils.Checkpointer(
        config.work_dir, model, None, None, logger)
    checkpointer.load(config.test_model_path, weights_only=True)
  else:
    if config.is_distributed:
      model = model.module
    torch.cuda.empty_cache()

  data_loader = get_data_loader(config)

  logger.info(f'Start testing on `{data_loader.dataset.name}` dataset with '
              f'{len(data_loader.dataset)} samples!')
  init_time = time()

  model.eval()
  predictions = {}
  for _, (images, _, image_ids) in enumerate(data_loader):
    images = images.to(GPU_DEVICE)
    image_ids = image_ids.tolist()
    with torch.no_grad():
      outputs = model(images)
      outputs = [output.to(CPU_DEVICE) for output in outputs]
    for image_id, output in zip(image_ids, outputs):
      predictions[image_id] = output
  dist_utils.synchronize()

  total_time = time() - init_time
  logger.info(f'Total inference time: {timedelta(seconds=int(total_time))} '
              f'({total_time / len(data_loader.dataset):.3f} sec / image).')

  all_predictions = dist_utils.gather(predictions)
  if not is_chef:
    dist_utils.synchronize()
    return

  image_ids = sorted(list(all_predictions))
  results = [all_predictions[image_id] for image_id in image_ids]
  assert len(results) == len(data_loader.dataset)
  results = torch.stack(results, dim=0)
  torch.save(results, os.path.join(config.work_dir, 'predictions.pth'))
  labels = torch.Tensor(data_loader.dataset.labels).type(torch.int64)
  acc_val = accuracy(results, labels[:, config.label_id], top_k=(1,))[0].item()
  logger.info(f'Accuracy on testing dataset is {acc_val:.2f}%.')

  dist_utils.synchronize()
