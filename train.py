import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import importlib
import torch.optim as optim
import torch.nn
import time
from util.ops_normal import *
from ops.config import Config

# Configuration

if __name__ == '__main__':
    config_path = "net_info.yml"
    config = Config(config_path)
    config.CONFIG_PATH = config_path

    if config.CUDA and not torch.cuda.is_available():
        raise Exception("[INFO] gpu is not available")

    # MODEL LOAD
    model = dynamic_model_load(config)
    data_loader = DataLoader(dynamic_dataloder_load(config), batch_size=config.BATCH_SIZE, shuffle=True)

    model.set_trainer(data_loader)
#    model.trainer.print_model_param()

    model.trainer.training()
