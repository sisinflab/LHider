import argparse
from src.jobs.merge import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', choices=['raw', 'clean', 'train'], default='clean')
    parser.add_argument('--score_type', choices=['manhattan', 'euclidean', 'jaccard'],
                        default='manhattan')
    parser.add_argument('--eps_rr', required=False, type=float, help='privacy budget of generated datasets')
    parser.add_argument('--eps_exp', required=False, type=float, help='exponential mechanism privacy budget')
    parser.add_argument('--output', required=False)

    arguments = vars(parser.parse_args())
    run(arguments)
