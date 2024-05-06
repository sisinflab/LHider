import argparse
from src.jobs.generate import run_generation
from src.jobs.split_generated import run as run_split
from src.jobs.recs import run as run_recs
from utils.collect_results import run as run_collect


def run(args):
    for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        args = {
            'dataset': args['dataset'],
            'dataset_name': args['dataset'],
            'dataset_type': 'train',
            'type': 'train',
            'eps_phi': eph_phi,
            'randomizer': 'randomized',
            'base_seed': args['base_seed'],
            'score_type': args['score_type'],
            'generations': args['generations'],
            'seed': args['seed']
        }
        # generate randomized datasets
        run_generation(args)
    # apply train-validation splitting on the generated datasets
    run_split(args)
    # compute recommendations on the randomized datasets
    run_recs(args)
    # collect the recommendation results in a unique folder
    run_collect(args)


if __name__ == '__main__':
    # user defined arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='jaccard')
    arguments = parser.parse_args()
    # fixed arguments
    arguments.generations = 500
    arguments.base_seed = 0
    arguments.seed = 0
    # run script
    run(vars(arguments))
