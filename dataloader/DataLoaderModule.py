from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.ColorizationDataLoaderUtil import *
import numpy as np
from skimage import io
import random

class DataLoaderModule(Dataset):
    def __init__(self, file_list, transform=None):
        """
        :param file_list(list): all file list

        """
        self.name = "DataLoader"
        self.step = 0

        if type(file_list) is str:
            self.file_list = [filename[: -1] for filename in open(file_list, 'r').readlines()][:5]
        else:
            self.file_list = file_list

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

    def reset(self):
        random.shuffle(self.file_list)

    def __getitem__(self, idx):
        img = io.imread(self.file_list[idx])
        print(self.file_list[idx])

        sample = make_colorization_sample(img)
        for key in sample:
            sample[key] = self.transform(sample[key])

        return sample

    def __len__(self):
        return len(self.file_list)
