import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DATA(Dataset):
  def __init__(self, path, shift=True):
    self.data = np.load(path)['X']
    self.len = self.data.shape[0]
    self.shifts = [0,10,20,30,40,50,60,70,80]
    self.shift = shift

  def __shift__(self, x):
    ind = np.random.choice(range(len(self.shifts)))
    x1 = np.pad(x, [[0,0],[0, 0],[self.shifts[ind],0]], constant_values=-1)[:,:,:128]
    return x1, ind

  def __getitem__(self, index):
    image = self.data[index]
    image = np.clip(image, -1,1)
    if self.shift:
      shifted, label = self.__shift__(image)
    else:
      label = 0
    return torch.from_numpy(image).float(), torch.from_numpy(shifted).float(), torch.Tensor([label]).int()

  def __len__(self):
    return self.len
