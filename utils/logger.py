# python3.7
"""Utility functions for logging."""

import os
import sys
import logging
from collections import OrderedDict

__all__ = ['setup_logger', 'SingleMetric', 'MetricsLogger']


def setup_logger(work_dir,
                 logfile_name='log.txt',
                 logger_name='logger',
                 distributed_rank=0):
  """Sets up logger from target work directory.

  Args:
    work_dir: The work directory. All intermediate files will be saved here.
    logfile_name: Name of the file to save log message. `log.txt` is used by
      default.
    logger_name: Name of the logger, which should be unique among different
      loggers. `logger` is used by default.
    distributed_rank: Rank of the current process in a distributed system. Only
      assign logger to the master process with Rank 0.

  Returns:
    A logger with `DEBUG` log level.

  Raises:
    SystemExit: If the work directory has already existed.
  """
  if distributed_rank:
    # Return an empty logger if the rank is not 0.
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    return logger

  if os.path.exists(work_dir):
    raise SystemExit(f'Work directory `{work_dir}` has already existed! '
                     f'Please specify another one.')
  os.makedirs(work_dir, exist_ok=True)

  logger = logging.getLogger(logger_name)
  if logger.hasHandlers():
    return logger
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

  # Print log message with `INFO` level or above onto the screen.
  sh = logging.StreamHandler(stream=sys.stdout)
  sh.setLevel(logging.INFO)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  # Save log message with all levels in log file.
  fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  return logger


class SingleMetric(object):
  """A class to record the status of a particular metric.

  Note that this class only supports numeral data type.
  """

  def __init__(self,
               name,
               log_format='.3f',
               log_type='AVERAGE',
               log_prefix='',
               log_tail=''):
    """Initializes with metric name and log format.

    Args:
      name: Name of the metric.
      log_format: This fields determines the log format.
      log_type: This field determines which kind of message will be printed.
        `AVERAGE` is used by default. `CURRENT`, `AVERAGE` and `CUMULATIVE` are
        supported.
      log_prefix: The prefix string for logging.
      log_tail: The tail string for logging.

    Raises:
      SystemExit: If the input `log_type` is not supported.
    """
    self.name = name
    self.log_format = log_format
    self.log_type = log_type.upper()
    self.log_prefix = log_prefix
    self.log_tail = log_tail
    if self.log_type not in ['CURRENT', 'AVERAGE', 'CUMULATIVE']:
      raise SystemExit(f'Log type `{self.log_type}` is not supported!')
    self.val = 0  # Current value.
    self.sum = 0  # Cumulative value.
    self.avg = 0  # Averaged value.
    self.cnt = 0  # Number of accumulations.

  def reset(self):
    """Resets the status."""
    self.val = 0
    self.sum = 0
    self.avg = 0
    self.cnt = 0

  def update(self, value):
    """Updates the status."""
    self.val = value
    self.cnt = self.cnt + 1
    self.sum = self.sum + value
    self.avg = self.sum / self.cnt

  def get_log_value(self):
    """Gets value according to log type."""
    if self.log_type == 'CURRENT':
      log_value = self.val
    elif self.log_type == 'AVERAGE':
      log_value = self.avg
    elif self.log_type == 'CUMULATIVE':
      log_value = self.sum
    else:
      raise NotImplementedError(
          f'Log type `{self.log_type}` is not implemented!')

    return log_value

  def __str__(self):
    string = (f'{self.log_prefix}'
              f'{self.name}: {self.get_log_value():{self.log_format}}'
              f'{self.log_tail}')
    return string


class MetricsLogger(object):
  """A class to log information of all available metrics."""

  def __init__(self, delimiter=', '):
    """Initializes with desired delimiter and an empty metric dictionary."""
    self.delimiter = delimiter
    self.metrics = OrderedDict()

  def add_metric(self, name, **kwargs):
    """Adds a new metric to the dictionary."""
    assert name not in self.metrics
    self.metrics[name.lower()] = SingleMetric(name, **kwargs)

  def reset(self, exclude_list=None):
    """Resets all metrics (if needed) inside the logger."""
    if exclude_list:
      exclude_list = [key.lower() for key in exclude_list]
    else:
      exclude_list = []
    exclude_list = set(exclude_list)
    for key in self.metrics.keys():
      if key not in exclude_list:
        self.metrics[key].reset()

  def update(self, **kwargs):
    """Updates the with any number of metrics."""
    for key, val in kwargs.items():
      assert isinstance(val, (float, int))
      assert key.lower() in self.metrics
      self.metrics[key.lower()].update(val)

  def __getattr__(self, name):
    if name.lower() in self.metrics:
      return self.metrics[name.lower()]
    if name in self.__dict__:
      return self.__dict__[name]
    raise AttributeError(
        f'`{type(self).__name__}` object has no attribute `{name}`!')

  def __str__(self):
    log_strings = []
    for _, metric in self.metrics.items():
      log_strings.append(str(metric))
    return self.delimiter.join(log_strings)
