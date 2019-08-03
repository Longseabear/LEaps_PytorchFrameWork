import numpy as np
import torch
import random
import os

def dynamic_model_load(config):
    module = __import__('%s' %(config.MODULE_NAME), fromlist=[config.MODULE_NAME])
    instance = getattr(module, config.CLASS_NAME)(config)
    return instance

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True