import argparse
from config_templates.training import TEMPLATE
from src.loader.paths import *
from elliot.run import run_experiment

config_dir = 'config_files/'
RANDOM_SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
args = parser.parse_args()

dataset_name = args.dataset

config = TEMPLATE.format(dataset=dataset_name)
config_path = os.path.join(config_dir, 'runtime_conf.yml')
with open(config_path, 'w') as file:
    file.write(config)
run_experiment(config_path)
