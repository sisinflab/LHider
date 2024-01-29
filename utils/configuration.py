import yaml
from src.loader.paths import *

def configuration_path(config_name):
    return os.path.join(CONFIG_DIR, config_name + '.yml')


def configuration(config_name):
    path = configuration_path(config_name)
    print(f'Reading configuration file from: \'{path}\'')
    with open(path, 'r') as file:
        conf = yaml.load(file, yaml.SafeLoader)
    return conf

print(configuration('parameters'))