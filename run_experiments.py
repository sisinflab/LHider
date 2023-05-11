import glob
import argparse
from config_templates.training import TEMPLATE
from src.loader.paths import *
from elliot.run import run_experiment

config_dir = 'config_files/'
RANDOM_SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--all',action='store_true')
args = parser.parse_args()


if args.all:
    dataset_list = [os.path.basename(absolute_path)
                    for absolute_path in glob.glob(os.path.join(RESULT_DIR, f'{args.dataset}*'))]
else:
    dataset_list = [f'{args.dataset}']

for dataset_name in dataset_list:
    config = TEMPLATE.format(dataset=dataset_name)
    config_path = os.path.join(config_dir, 'runtime_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)
