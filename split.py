import argparse
from src.jobs.split_generated import run as run_split


def run(args):
    args = {
        'dataset_name': args['dataset'],
        'dataset_type': 'train',
        'base_seed': args['base_seed'],
    }
    # apply train-validation splitting on the generated datasets
    run_split(args)


if __name__ == '__main__':
    # user defined arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    arguments = parser.parse_args()
    # fixed arguments
    arguments.base_seed = 0
    # run script
    run(vars(arguments))
