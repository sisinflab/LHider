import argparse
from src.jobs.subsample import run_subsampled
from src.jobs.split_generated import run as run_split
from src.jobs.recs import run as run_recs
from utils.collect_results import run as run_collect

base_seed = 0


def run(args):
    seed = base_seed
    for eps_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        for reps in [1, 10, 100, 1000]:
            eps_exp = [0.001, 0.002, 0.004, 0.008]

            # run
            args = {
                'dataset': args['dataset'],
                'type': 'train',
                'dataset_name': args['dataset'],
                'dataset_type': 'train',
                'eps_phi': eps_phi,
                'randomizer': 'randomized',
                'reps': reps,
                'eps_exp': eps_exp,
                'seed': seed,
                'base_seed': base_seed,
                'score_type': args['score_type']
            }

            run_subsampled(args)
            seed += len(eps_exp)

    run_split(args)
    run_recs(args)
    run_collect(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    arguments = parser.parse_args()
    run(vars(arguments))
