# python3.7
"""Contains testing tools."""

import os.path
from time import time
from datetime import timedelta

import pickle
import torch

# pylint: disable=import-error, no-name-in-module
from configs import CPU_DEVICE, GPU_DEVICE
from models.model_handler import get_model
from data.loader import get_data_loader
import utils.checkpoint as ckpt_utils
import utils.distribute as dist_utils
# pylint: enable=import-error, no-name-in-module

__all__ = ['test']


def test(config, logger, model=None):
  """Model testing."""
  is_chef = (dist_utils.get_rank() == 0)  # pylint: disable=unused-variable

  if model is None:
    logger.info(f'Deploy model.')
    model = get_model(model_name=config.model_structure,
                      resolution=config.input_size)
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
  for _, (images, _, image_ids) in enumerate(data_loader):
    images = images.to(GPU_DEVICE)
    image_ids = image_ids.tolist()
    with torch.no_grad():
      _, _, outputs = model.encode(images)
      outputs = [output.to(CPU_DEVICE) for output in outputs]
      outputs = list(zip(*outputs))
      for image_id, output in zip(image_ids, outputs):
        assert len(output) == len(model.strides)
        image_name = data_loader.dataset.images[image_id]
        result = {'image_name': image_name}
        res = model.resolution
        for code_idx, code in enumerate(output):
          code = code.detach().numpy()
          res = res // model.strides[code_idx]
          assert code.shape == (res, res)
          result[f'code{code_idx}'] = code
        save_name = os.path.splitext(image_name)[0] + '.pkl'
        with open(os.path.join(config.work_dir, save_name), 'wb') as f:
          pickle.dump(result, f)
  dist_utils.synchronize()

  total_time = time() - init_time
  logger.info(f'Total inference time: {timedelta(seconds=int(total_time))} '
              f'({total_time / len(data_loader.dataset):.3f} sec / image).')
