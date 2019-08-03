import torch
from torch.utils.data import DataLoader
from dataloader.DataLoaderModule import DataLoaderModule
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import importlib
import torch.optim as optim
import time
from model.Net import Net
from util.ops_normal import *
from ops.config import Config

# Configuration

if __name__ == '__main__':
    # INIT for deterministic process
    config = Config("net_info.yml")
    print(config)

    # MODEL LOAD
    model = dynamic_model_load(config)
    print(model)

    data_loader = DataLoader(DataLoaderModule(config.FILE_LIST_PATH), batch_size=5, shuffle=True)
    model.set_trainer(data_loader)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, model.state_dict()[param_tensor].size())

    model.trainer.training()
