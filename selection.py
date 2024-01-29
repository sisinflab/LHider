import argparse
from src.jobs.exponential import run as run_select

n = 500
folder = 0
seed = 0
dimensions = [10, 30, 50, 75, 100, 1000]


def run(args):
    args = {
        'dataset': args['dataset'],
        'dataset_name': args['dataset'],
        'dataset_type': 'train',
        'type': 'train',
        'randomizer': 'randomized',
        'base_seed': folder,
        'score_type': args['score_type'],
        'generations': n,
        'seed': seed,
        'dimensions': dimensions
    }
    run_select(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    arguments = parser.parse_args()
    run(vars(arguments))



