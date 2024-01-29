import argparse
from src.jobs.sigir_itemknn import run_generation
from src.jobs.split_generated import run as run_split
from src.jobs.recs import run as run_recs
from utils.collect_results import run as run_collect

n = 2
folder = 0
seed = 0


def run(args):
    for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        args = {
            'dataset': args['dataset'],
            'dataset_name': args['dataset'],
            'dataset_type': 'train',
            'type': 'train',
            'eps_phi': eph_phi,
            'randomizer': 'randomized',
            'base_seed': folder,
            'score_type': args['score_type'],
            'generations': n,
            'seed': seed
        }
        run_generation(args)
    run_split(args)
    run_recs(args)
    run_collect(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    arguments = parser.parse_args()
    run(vars(arguments))

