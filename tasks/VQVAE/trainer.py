# python3.7
"""Contains training tools."""

import os.path
from time import time
from datetime import timedelta

import numpy as np
import cv2
import torch
from tensorboardX import SummaryWriter

# pylint: disable=import-error, no-name-in-module
from configs import GPU_DEVICE
from models.model_handler import get_model
from data.loader import get_data_loader
from tools.losses import get_loss
from tasks.VQVAE.tester import test
import utils.checkpoint as ckpt_utils
import utils.distribute as dist_utils
import utils.logger as log_utils
import utils.scheduler as sched_utils
# pylint: enable=import-error, no-name-in-module

__all__ = ['train']


def train(config, logger):
  """Model training."""
  is_chef = (dist_utils.get_rank() == 0)

  logger.info(f'Deploy model.')
  model = get_model(model_name=config.model_structure,
                    resolution=config.input_size)
  model.to(GPU_DEVICE)
  if config.is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        device_ids=[config.local_rank],
        output_device=config.local_rank)

  logger.info(f'Get training strategy.')
  optimizer = sched_utils.get_optimizer(config, model)
  lr_scheduler = sched_utils.get_lr_scheduler(config, optimizer)

  checkpointer = ckpt_utils.Checkpointer(
      config.work_dir, model, optimizer, lr_scheduler, logger)
  if config.load_path:
    ckpt_data = checkpointer.load(config.load_path, config.load_weights_only)
  else:
    ckpt_data = {}
  start_step = ckpt_data.pop('step') if 'step' in ckpt_data else 0

  data_loader = get_data_loader(config, start_step=start_step)

  meters = log_utils.MetricsLogger()
  meters.add_metric('Time', log_format='.3f', log_tail=' sec')  # Step time
  meters.add_metric('Data', log_format='.3f', log_tail=' sec')  # Data time
  meters.add_metric('Loss', log_format='.3e')  # Total loss
  meters.add_metric('Rec_Loss', log_format='.3e')  # Reconstruction loss
  meters.add_metric('Commit_Loss', log_format='.3e')  # Commitment loss
  if not config.disable_tensorboard and is_chef:
    summary_writer = SummaryWriter(config.work_dir)
  else:
    summary_writer = None

  logger.info(f'Start training on `{data_loader.dataset.name}` dataset with '
              f'{len(data_loader.dataset)} samples!')
  model.train()
  init_time = time()
  end_time = time()
  for step, (images, _, _) in enumerate(data_loader, start_step + 1):
    data_time = time() - end_time
    lr_scheduler.step()

    images = images.to(GPU_DEVICE)
    outputs, commitment_loss = model(images)
    losses = {
        'reconstruction': get_loss('l2')(outputs, images),
        'commitment': commitment_loss,
    }

    total_loss = sum(loss for loss in losses.values())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses_reduced = dist_utils.reduce(losses)
    total_loss_val = sum(loss for loss in losses_reduced.values()).item()
    rec_loss_val = losses_reduced['reconstruction'].item()
    commit_loss_val = losses_reduced['commitment'].item()
    meters.update(loss=total_loss_val,
                  rec_loss=rec_loss_val,
                  commit_loss=commit_loss_val)
    if summary_writer:
      summary_writer.add_scalar('Loss/total_loss', total_loss_val, step)
      summary_writer.add_scalar('Loss/reconstruction_loss', rec_loss_val, step)
      summary_writer.add_scalar('Loss/commitment_loss', commit_loss_val, step)

    step_time = time() - end_time
    meters.update(time=step_time, data=data_time)
    eta = timedelta(seconds=int(meters.time.avg * (config.max_step - step)))

    if step % config.log_interval == 0 or step == config.max_step:
      lr = optimizer.param_groups[0]['lr']
      logger.info(f'Step {step:7d} (lr={lr:.2e}): {meters}. (ETA: {eta})')
      meters.reset()
    if is_chef and (
        step == 1 or step % config.save_step == 0 or step == config.max_step):
      checkpointer.save(f'model-{step:07d}.pth', step=step)
    if is_chef and step == config.max_step:
      checkpointer.save(f'model-final.pth', step=step)

    viz_step = int(getattr(config, 'viz_step', 1000))
    viz_num = int(getattr(config, 'viz_num', 10))

    if is_chef and step % viz_step == 0:
      viz_num = min(viz_num, config.batch_size_per_gpu)
      size = config.input_size
      viz_image = np.zeros((2 * size, viz_num * size, 3), np.uint8)
      with torch.no_grad():
        real_images = images[:viz_num].cpu().detach().numpy()
        real_images = (real_images + 1) * 255 / 2.0
        real_images = np.clip(real_images + 0.5, 0, 255).astype(np.uint8)
        real_images = real_images.transpose(0, 2, 3, 1)
        rec_images = outputs[:viz_num].cpu().detach().numpy()
        rec_images = (rec_images + 1) * 255 / 2.0
        rec_images = np.clip(rec_images + 0.5, 0, 255).astype(np.uint8)
        rec_images = rec_images.transpose(0, 2, 3, 1)
        for i in range(viz_num):
          viz_image[:size, i * size:(i + 1) * size] = real_images[i, :, :, ::-1]
          viz_image[size:, i * size:(i + 1) * size] = rec_images[i, :, :, ::-1]
        cv2.imwrite(os.path.join(config.work_dir, f'step_{step:07d}.jpg'),
                    viz_image)

    end_time = time()

  total_time = time() - init_time
  training_step = config.max_step - start_step
  logger.info(f'Total training time: {timedelta(seconds=int(total_time))} '
              f'({total_time / training_step:.3f} sec / step).')
  logger.info(f'------------------------------')

  if not config.skip_final_test:
    config.run_mode = 'test'
    config.work_dir = os.path.join(config.work_dir, 'final_test')
    if is_chef:
      os.makedirs(config.work_dir)
    test(config, logger, model)
