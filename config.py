import os


class Config(object):
    root = os.path.abspath(os.path.dirname(__file__))
    scratch_root = os.path.abspath(os.path.dirname(__file__))
    group_root = os.path.abspath(os.path.dirname(__file__))
    dir_shared = os.path.join(root, 'tmp')
    backend = 'gloo'


class LocalConfig(Config):
    name = 'local'


class RoarConfig(Config):
    name = 'roar'
    scratch_dir = '/storage/home/kms8227/scratch'
    scratch_root = os.path.join(scratch_dir, 'ffa-p2-priv')
    group_dir = '/storage/group/kms8227/default'
    group_root = os.path.join(group_dir, 'ffa-p2-priv')


CONFIGS = {
    'local': LocalConfig,
    'roar': RoarConfig,
}

def load_config(config_name):
    return CONFIGS[config_name]()
