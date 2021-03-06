# python3.7
"""Contains customized transforms."""

import torchvision.transforms as transforms

__all__ = ['get_transform']


def simple_classification_transforms(config):
  """Defines the simple transformation for classification task.

  Basically, it pre-processes the image with following steps:
  (1) Randomly flip the image horizontally (for training only).
  (2) Randomly crop the image (for training only).
  (3) Convert the image to torch.Tensor with range [0, 1] and in `CHW` format.
  (4) Normalize the image.
  """
  if config.run_mode == 'train':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(size=config.input_size,
                                     scale=(0.9, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
  else:
    transform = transforms.Compose([
        transforms.Resize(size=config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
  return transform


def simple_generation_transforms(config):
  """Defines the simple transformation for image generation task.

  Basically, it pre-processes the image with following steps:
  (1) Randomly flip the image horizontally.
  (2) Crop the image.
  (3) Convert the image to torch.Tensor with range [0, 1] and in `CHW` format.
  (4) Normalize the image.
  """
  flip_rate = 0.5 if config.run_mode == 'train' else 0.0
  transform = transforms.Compose([
      transforms.RandomHorizontalFlip(flip_rate),
      transforms.Resize(size=config.input_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])
  return transform


TRANSFORMS = {
    'simple_classification': simple_classification_transforms,
    'simple_generation': simple_generation_transforms,
}


def get_transform(transform_name, config, **kwargs):
  """Gets transform by name."""
  transform_name = transform_name.lower()
  try:
    transform = TRANSFORMS[transform_name](config, **kwargs)
  except KeyError:
    raise ValueError(f'Transform `{transform_name}` is not supported!\n'
                     f'Please choose from {list(TRANSFORMS)}.')
  return transform
