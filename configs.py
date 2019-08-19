# python3.7
"""Contains the configurations used when running a model."""

import argparse

__all__ = ['get_config', 'CPU_DEVICE', 'GPU_DEVICE']

CPU_DEVICE = 'cpu'
GPU_DEVICE = 'cuda'


def get_config(current_environ):
  """Gets configurations by parsing arguments.

  This function will also do some critical checks. Environ

  Args:
    current_environ: This field contains current environment variables.

  Returns:
    All configurations for running model.

  Raises:
    SystemExit: If any of the checks failed.
  """
  parser = argparse.ArgumentParser(description='Run model with PyTorch.')

  parser.add_argument('--local_rank', type=int, default=0,
                      help='This field, indicating the process rank on current '
                           'node, is required by `torch.distributed.launch`. '
                           'Please do not modify.')

  parser.add_argument('--work_dir', type=str, required=True,
                      help='Work directory, where all results will be saved.')
  parser.add_argument('--run_mode', type=str, default='train',
                      choices=['train', 'test'],
                      help='Running mode, which can only be `train` or `test`.')
  parser.add_argument('--skip_final_test', default=False, action='store_true',
                      help='Whether to skip final test at training mode.')

  parser.add_argument('--task_folder', required=True,
                      help='Folder that contains the running scripts for '
                           'target task.')

  parser.add_argument('--model_structure', type=str, required=True,
                      help='Structure of the model to deploy.')
  parser.add_argument('--use_pretrain', default=False, action='store_true',
                      help='Whether to use pre-trained weights.')
  parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes for classification task.')
  parser.add_argument('--label_id', type=int, default=0,
                      help='Which label to use among a list of labels.')

  parser.add_argument('--load_path', type=str, default='',
                      help='Path to the checkpoint (or directory) to resume '
                           'training or perform testing.')
  parser.add_argument('--load_weights_only', default=False, action='store_true',
                      help='Whether to only load weights w/o optimizer and '
                           'learning rate.')
  parser.add_argument('--test_model_path', type=str, default='',
                      help='Path to the checkpoint (or directory) for testing.')
  parser.add_argument('--disable_tensorboard', default=False,
                      action='store_true',
                      help='Whether to disable tensorboard summary writer.')

  parser.add_argument('--max_step', type=int, default=100000,
                      help='Max running step. For training only.')
  parser.add_argument('--log_interval', type=int, default=100,
                      help='Step interval to log message. For training only.')
  parser.add_argument('--save_times', type=int, default=10,
                      help='Number of checkpoints to save. For training only.')

  parser.add_argument('--lr_base', type=float, default=0.01,
                      help='Base learning rate.')
  parser.add_argument('--lr_bias_factor', type=float, default=2.0,
                      help='Multiplier for `bias` parameters.')
  parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Weight decay for `weight` parameters.')
  parser.add_argument('--weight_decay_bias', type=float, default=0.0,
                      help='Weight decay for `bias` parameters.')
  parser.add_argument('--optimizer_type', type=str, default='sgd',
                      help='Optimizer type.')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum factor for SGD optimizer.')
  parser.add_argument('--lr_steps', type=str, default='',
                      help='Learning rate decay steps, joined with `,`.')
  parser.add_argument('--lr_decay', type=float, default=0.1,
                      help='Learning rate decay factor.')
  parser.add_argument('--lr_warmup_factor', type=float, default=0.0,
                      help='Initial warm-up factor for learning rate.')
  parser.add_argument('--lr_warmup_steps', type=int, default=500,
                      help='Learning rate warm-up steps.')

  parser.add_argument('--train_dataset_name', type=str, default='',
                      help='Name of the training dataset used.')
  parser.add_argument('--train_image_dir', type=str, default='/',
                      help='Directory where the training images saved.')
  parser.add_argument('--train_label_file', type=str, default='',
                      help='Path to the label file of the training dataset.')
  parser.add_argument('--test_dataset_name', type=str, default='',
                      help='Name of the testing dataset used.')
  parser.add_argument('--test_image_dir', type=str, default='/',
                      help='Directory where the testing images saved.')
  parser.add_argument('--test_label_file', type=str, default='',
                      help='Path to the label file of the testing dataset.')
  parser.add_argument('--data_transform', type=str, default='',
                      help='Name of the data transform.')
  parser.add_argument('--input_size', type=int, default=224,
                      help='Size of the input image.')
  parser.add_argument('--batch_size_per_gpu', type=int, default=1,
                      help='Batch size assigned to each GPU.')
  parser.add_argument('--num_workers_per_gpu', type=int, default=1,
                      help='Number of works to load data on each GPU.')

  config = parser.parse_args()

  if not config.work_dir:
    raise SystemExit(f'Work directory should be specified!')
  if not config.model_structure:
    raise SystemExit(f'Model structure should be specified!')
  if config.run_mode == 'train' and not config.train_dataset_name:
    raise SystemExit(f'Dataset should be specified at `train` mode!')
  if (config.run_mode == 'train' and config.skip_final_test and
      not config.test_dataset_name):
    raise SystemExit(f'Dataset should be specified at `train` mode '
                     f'with final test!')
  if config.run_mode == 'test' and not config.test_dataset_name:
    raise SystemExit(f'Dataset should be specified at `test` mode!')
  if not config.data_transform:
    raise SystemExit(f'Data transform should be specified!')
  if config.run_mode == 'test' and not config.test_model_path:
    raise SystemExit(f'Test model path should be specified in `test` mode!')
  if config.batch_size_per_gpu < 1:
    raise SystemExit(f'Batch size per GPU should be a positive integer, '
                     f'but {config.batch_size_per_gpu} received!')
  if config.num_workers_per_gpu < 1:
    config.num_workers_per_gpu = 1

  if 'WORLD_SIZE' in current_environ:
    config.num_gpus = int(current_environ['WORLD_SIZE'])
  else:
    config.num_gpus = 1
  config.is_distributed = (config.num_gpus > 1)

  config.save_step = config.max_step // config.save_times
  if not config.lr_steps:
    config.lr_steps = []
  else:
    config.lr_steps = sorted(list(map(int, config.lr_steps.split(','))))

  return config
