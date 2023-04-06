import os
import argparse
from config_template import TEMPLATE
from elliot.run import run_experiment

config_dir = 'config_files/'
RANDOM_SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--eps', default=-1, type=float, nargs='+')
args = parser.parse_args()

epsilon = args.eps

dataset_list = []
if epsilon != -1:
    dataset_list = [f'{args.dataset}_eps{eps}' for eps in epsilon]
else:
    dataset_list = [f'{args.dataset}']

for dataset_name in dataset_list:
    config = TEMPLATE.format(dataset=dataset_name)
    config_path = os.path.join(config_dir, 'runtime_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)
