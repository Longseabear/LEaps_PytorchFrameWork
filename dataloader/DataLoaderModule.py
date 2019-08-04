from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import random
from skimage.color import rgb2lab, rgb2gray
import numpy as np
from PIL import Image
import torch

class BaseLoader(Dataset):
    def __init__(self, config, file_list):
        """
        :param file_list(list): all file list

        """
        self.name = "DataLoader"
        self.step = 0
        self.config = config

        if type(file_list) is str:
            self.file_list = [filename[:-1] for filename in open(file_list, 'r').readlines()]
        else:
            self.file_list = file_list

        if config.DATASET_SIZE is not -1:
            self.file_list = self.file_list[:config.DATASET_SIZE]

    def reset(self):
        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)


class ColorizationDataLoader(BaseLoader):
    def __init__(self, config):
        super(ColorizationDataLoader, self).__init__(config, config.FILE_LIST_PATH)
        """
        :param file_list(list): all file list

        """
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.hint_density = config.HINT_DENSITY

    def make_colorization_sample(self, img):
        h, w, c = img.shape

        lab = rgb2lab(img)
        hint_mask = np.random.random([h, w, 1])
        hint_mask[hint_mask > self.hint_density] = 0
        hint_mask[hint_mask != 0] = 1

        sample = {'lab': lab, 'hint': hint_mask}
        return sample

    def __getitem__(self, idx):
        img = io.imread(self.file_list[idx])
        sample = self.make_colorization_sample(img)

        for key in sample:
            sample[key] = self.transform(sample[key]).type(torch.FloatTensor)

        return sample

