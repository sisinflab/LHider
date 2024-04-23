import argparse
from src.jobs.generate import run_generation
from src.jobs.split_generated import run as run_split
from src.jobs.recs import run as run_recs
from utils.collect_results import run as run_collect

n = 100
folder = 0
seed = 0


def run(args):
    for eph_phi in [1, 2, 4, 8]:
        args = {
            'dataset': args['dataset'],
            'dataset_name': args['dataset'],
            'dataset_type': 'train',
            'type': 'train',
            'eps_phi': eph_phi,
            'randomizer': args['randomizer'],
            'base_seed': folder,
            'score_type': args['score_type'],
            'generations': n,
            'seed': seed,
            'min_val': args['min_val'],
            'max_val': args['max_val']
        }
        run_generation(args)
    # run_split(args)
    # run_recs(args)
    # run_collect(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    parser.add_argument('--min_val', type=int)
    parser.add_argument('--max_val', type=int)
    parser.add_argument('--randomizer', default='subsampled')
    arguments = parser.parse_args()
    run(vars(arguments))

