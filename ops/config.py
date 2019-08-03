import yaml
import os


class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    def __init__(self, file_path):
        super(Config, self).__init__()
        assert os.path.exists(file_path), "FILE NOT FOUND ERROR: Config File doesn't exist. : {}".format(file_path)
        try:
            with open(file_path, 'r') as f:
                self.member = yaml.load(f)
        except:
            assert False

        # self.LOGGER_PATH = self.MODEL_PATH + self.LOGGER_PATH
        # self.CHECKPOINT_PATH = self.MODEL_PATH + self.CHECKPOINT_PATH
        # self.TENSORBOARD_LOG_PATH = self.MODEL_PATH + self.TENSORBOARD_LOG_PATH
        #
        # os.makedirs(self.LOGGER_PATH,exist_ok=True)
        # os.makedirs(self.CHECKPOINT_PATH,exist_ok=True)
        # os.makedirs(self.PRETRAINED_MODEL_PATH,exist_ok=True)
        # os.makedirs(self.TENSORBOARD_LOG_PATH,exist_ok=True)

    def __getattr__(self, name):
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

    def __str__(self):
        return str(self.member)