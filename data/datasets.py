# python3.7
"""Contains various datasets."""

import os.path
from PIL import Image

from torch.utils.data import Dataset

__all__ = ['get_dataset']


class CelebAHQDataset(Dataset):
  """Defines CelebA-HQ dataset."""

  def __init__(self, image_dir, label_file, transforms=None):
    super().__init__()
    self.name = 'celebahq'
    self.image_dir = image_dir

    self.images = []
    self.labels = []
    with open(label_file, 'r') as f:
      self.attr_list = f.readline().strip().split(' ')
      for _, line in enumerate(f):
        line = line.strip().split(' ')
        assert len(line) == len(self.attr_list) + 1
        self.images.append(line[0])
        self.labels.append(list(map(int, line[1:])))
    assert len(self.images) == len(self.labels)

    self.transforms = transforms

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    image_path = os.path.join(self.image_dir, self.images[index])
    image = Image.open(image_path).convert('RGB')
    label = self.labels[index]

    # Only perform transforms on images.
    if self.transforms is not None:
      image = self.transforms(image)

    return image, label, index


DATASETS = {
    'celebahq': CelebAHQDataset,
}


def get_dataset(dataset_name, **kwargs):
  """Gets dataset by name."""
  dataset_name = dataset_name.lower()
  try:
    dataset = DATASETS[dataset_name](**kwargs)
  except KeyError:
    raise ValueError(f'Dataset `{dataset_name}` is not supported!\n'
                     f'Please choose from {list(DATASETS)}.')
  return dataset
