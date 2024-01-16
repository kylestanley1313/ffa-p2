import os


class Config(object):
    root = os.path.abspath(os.path.dirname(__file__))
    scratch_root = os.path.abspath(os.path.dirname(__file__))
    debug = False


class LocalDebugConfig(Config):
    debug = True


class RoarConfig(Config):
    scratch_root = '/storage/home/kms8227/scratch/ffa-p2-priv'


class RoarDebugConfig(Config):
    scratch_root = '/storage/home/kms8227/scratch/ffa-p2-priv'
    debug = True


CONFIGS = {
    'local': Config,
    'local_debug': LocalDebugConfig,
    'roar': RoarConfig,
    'roar_debug': RoarDebugConfig
}


def load_config(config_name):
    return CONFIGS[config_name]()
