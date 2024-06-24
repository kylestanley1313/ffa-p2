import os


class Config(object):
    root = os.path.abspath(os.path.dirname(__file__))
    scratch_root = os.path.abspath(os.path.dirname(__file__))
    dir_shared = os.path.join(root, 'tmp')
    backend = 'gloo'


class LocalConfig(Config):
    pass


class RoarConfig(Config):
    scratch_root = '/storage/home/kms8227/scratch/ffa-p2-priv'



CONFIGS = {
    'local': LocalConfig,
    'roar': RoarConfig,
}

def load_config(config_name):
    return CONFIGS[config_name]()
