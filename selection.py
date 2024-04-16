import argparse
from src.jobs.exponential import run as run_select

folder = 0
seed = 0
dimensions = [1, 2, 4, 8, 16, 32, 64]


def run(args):
    args = {
        'dataset': args['dataset'],
        'dataset_name': args['dataset'],
        'dataset_type': 'train',
        'type': 'train',
        'randomizer': args['randomizer'],
        'base_seed': folder,
        'score_type': args['score_type'],
        'seed': seed,
        'dimensions': dimensions,
        'metrics': ['LogReg_F1']
    }
    run_select(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    parser.add_argument('--randomizer', default='randomized')
    arguments = parser.parse_args()
    run(vars(arguments))



