import argparse
from src.jobs.lhider import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='name of the dataset that is going to be anonymized')
    parser.add_argument('--type', choices=['raw', 'clean', 'train', 'val', 'test'], default='clean')
    parser.add_argument('--score_type', choices=['manhattan', 'euclidean', 'cosineUser', 'cosineItem', 'jaccard'], default='manhattan')
    parser.add_argument('--eps_rr', required=True, type=float, help='privacy budget of generated datasets')
    parser.add_argument('--eps_exp', required=False, type=float, nargs='+', default=[1],
                        help='exponential mechanism privacy budget')
    parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')

    args = vars(parser.parse_args())
    run(args)
