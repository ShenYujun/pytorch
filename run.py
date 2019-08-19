# python3.7
"""Contains the main script to run model.

Note that this script is better to work with following command

python -m torch.distributed.launch \
       --nnodes=$NUMBER_OF_NODES_TO_USE \
       --node_rank=$RANK_OF_CURRENT_NODE \
       --nproc_per_node=$NUMBER_OF_GPUS_TO_USE_ON_EACH_NODE \
       run.py --$OTHER_ARGS

More details can be found at file `torch/distributed/launch.py`.
"""

import os
import sys
import torch

from configs import get_config
from utils.logger import setup_logger
from utils.distribute import get_rank, get_env_info


def main():
  """Main function to run model."""
  config = get_config(os.environ)

  sys.path.append(os.path.join('tasks', config.task_folder))
  # pylint: disable=import-error
  from trainer import train
  from tester import test
  # pylint: enable=import-error

  if config.is_distributed:
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

  logger = setup_logger(config.work_dir, distributed_rank=get_rank())
  logger.info(f'Using {config.num_gpus} GPUs.')
  logger.info(f'Collecting environment info:{get_env_info()}')
  logger.info(f'------------------------------')
  logger.info(f'Running configurations:')
  for key, val in config.__dict__.items():
    logger.info(f'  {key}: {val}')
  logger.info(f'------------------------------')

  if config.run_mode == 'train':
    train(config, logger)
  elif config.run_mode == 'test':
    test(config, logger)


if __name__ == '__main__':
  main()
