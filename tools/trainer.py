# python3.7
"""Contains training tools."""

from time import time
from datetime import timedelta

import torch
from tensorboardX import SummaryWriter

from configs import GPU_DEVICE
from models.model_handler import get_model
from data.loader import get_data_loader
from tools.losses import get_loss, accuracy
from tools.tester import test
import utils.checkpoint as ckpt_utils
import utils.distribute as dist_utils
import utils.logger as log_utils
import utils.scheduler as sched_utils

__all__ = ['train']


def train(config, logger):
  """Model training."""
  is_chef = (dist_utils.get_rank() == 0)

  logger.info(f'Deploy model.')
  model = get_model(model_name=config.model_structure,
                    use_pretrain=config.use_pretrain,
                    num_classes=config.num_classes)
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
  meters.add_metric('Loss', log_format='.3f')  # Total loss
  meters.add_metric('Acc', log_format='.2f', log_tail='%')   # Accuracy
  if not config.disable_tensorboard and is_chef:
    summary_writer = SummaryWriter(config.work_dir)
  else:
    summary_writer = None

  logger.info(f'Start training on `{data_loader.dataset.name}` dataset with '
              f'{len(data_loader.dataset)} samples!')
  model.train()
  init_time = time()
  end_time = time()
  for step, (images, labels, _) in enumerate(data_loader, start_step + 1):
    data_time = time() - end_time
    lr_scheduler.step()

    images = images.to(GPU_DEVICE)
    labels = [label.to(GPU_DEVICE) for label in labels]
    outputs = model(images)
    losses = {
        'softmax': get_loss('softmax')(outputs, labels[config.label_id])
    }
    acc = {
        'top_1': accuracy(outputs, labels[config.label_id], top_k=(1,))[0]
    }

    total_loss = sum(loss for loss in losses.values())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses_reduced = dist_utils.reduce(losses)
    acc_reduced = dist_utils.reduce(acc)
    total_loss_val = sum(loss for loss in losses_reduced.values()).item()
    acc_1_val = acc_reduced['top_1'].item()
    meters.update(loss=total_loss_val, acc=acc_1_val)
    if summary_writer:
      summary_writer.add_scalar('Loss/total_loss', total_loss_val, step)
      summary_writer.add_scalar('Accuracy/top-1', acc_1_val, step)

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

    end_time = time()

  total_time = time() - init_time
  training_step = config.max_step - start_step
  logger.info(f'Total training time: {timedelta(seconds=int(total_time))} '
              f'({total_time / training_step:.3f} sec / step).')
  logger.info(f'------------------------------')

  if not config.skip_final_test:
    config.run_mode = 'test'
    test(config, logger, model)
