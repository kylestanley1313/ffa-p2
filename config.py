import os


class Config(object):

    root = os.path.abspath(os.path.dirname(__file__))
    scratch_root = os.path.abspath(os.path.dirname(__file__))
    dir_shared = os.path.join(root, 'tmp')
    
    backend = 'gloo'

    benchmark = False


class LocalBenchmarkConfig(Config):
    benchmark = True


class RoarConfig(Config):
    scratch_root = '/storage/home/kms8227/scratch/ffa-p2-priv'


class RoarBenchmarkConfig(Config):
    scratch_root = '/storage/home/kms8227/scratch/ffa-p2-priv'
    benchmark = True


CONFIGS = {
    'local': Config,
    'local_benchmark': LocalBenchmarkConfig,
    'roar': RoarConfig,
    'roar_benchmark': RoarBenchmarkConfig
}


def load_config(config_name):
    return CONFIGS[config_name]()
