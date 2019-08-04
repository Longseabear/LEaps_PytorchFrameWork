import numpy as np
import torch
import random
import os
from ops.config import Config
from shutil import copyfile


def dynamic_model_load(config):
    module = __import__('%s' % (config.MODULE_NAME), fromlist=[config.MODULE_NAME])
    instance = getattr(module, config.CLASS_NAME)(config)
    return instance

def dynamic_dataloder_load(config):
    module = __import__('%s' % (config.DATA_LOADER_MODULE_NAME), fromlist=[config.DATA_LOADER_MODULE_NAME])
    instance = getattr(module, config.DATA_LOADER_CLASS_NAME)(config)
    return instance

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_key_value_yaml(s):
    key = s[:s.find(":")].strip()
    val = s[s.find(":") + 1:].strip()
    return key, val


def modify_yaml(src, dst, config):
    res = ""
    with open(src, 'r') as fp:
        modify_mode = False
        for line in fp:
            if line.startswith("# *[end]"):
                modify_mode = False

            if modify_mode:
                key, val = get_key_value_yaml(line)
                new_val = config[key]
                if isinstance(new_val, str):
                    new_val = "'" + new_val + "'"
                new_line = key + ": " + str(new_val) + "\n"
                res += new_line
                continue

            if line.startswith("# *[auto]"):
                modify_mode = True

            res += line

    with open(dst, 'w+') as fp:
        fp.write(res)


def get_original_model_name(current_model_name):
    model_name = current_model_name
    if str.rfind(model_name, '_') is not -1:
        model_name = model_name[:str.rfind(model_name, '_')]
    return model_name
