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
                super().__setattr__('member', yaml.load(f))
        except:
            assert False

    def __getitem__(self, key):
        return self.member[key]

    def __setitem__(self, key, value):
        self.member[key] = value

    def set_inst_attr(self, attr, val):
        if attr == 'member':
            raise Exception("attr exception")
        super().__setattr__(attr,val)

    def __setattr__(self, attr, val):
        if attr in vars(self):
            self.set_inst_attr(attr,val)
        else:
            self.__setitem__(attr,val)

    def __getattr__(self, name):
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

    def __str__(self):
        return str(self.member)