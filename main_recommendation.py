import argparse
from config_templates.training import TEMPLATE
from src.loader.paths import *
from elliot.run import run_experiment

config_dir = 'config_files/'
RANDOM_SEED = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', choices=['raw', 'clean', 'train'], default='clean')
    parser.add_argument('--eps_rr', required=True, type=float, help='privacy budget of generated datasets')
    parser.add_argument('--eps_exp', required=True, type=float, help='exponential mechanism privacy budget')
    parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')
    args = parser.parse_args()

    dataset_name = f'{args.dataset}_{args.type}_epsrr{args.eps_rr}_epsexp{args.eps_exp}'

    # Writing the config file for the best model
    config = TEMPLATE.format(dataset=args.dataset, file=dataset_name)
    config_path = os.path.join(config_dir, 'runtime_conf.yml')

    with open(config_path, 'w') as file:
        file.write(config)

    run_experiment(config_path)
