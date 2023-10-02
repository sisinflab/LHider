import argparse
from src.jobs.recommendations import run
from config_templates.training import TEMPLATE
from src.loader.paths import *
from elliot.run import run_experiment

"""
This script runs the recommendations for the given dataset with the recommendation models in the configuration file
If the original dataset is selected, recommendations will be computed on the non-synthetic dataset
"""

config_dir = 'config_files/'
RANDOM_SEED = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--type', choices=['raw', 'clean', 'train'], default='clean')
    parser.add_argument('--score_type', choices=['manhattan', 'euclidean', 'cosineUser', 'cosineItem', 'jaccard'], default='manhattan')
    parser.add_argument('--eps_rr', required=False, type=float, help='privacy budget of generated datasets')
    parser.add_argument('--eps_exp', required=False, type=float, help='exponential mechanism privacy budget')
    parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')
    parser.add_argument('--mail', action='store_true')
    args = parser.parse_args()

    if args.mail:
        from email_notifier.email_sender import EmailNotifier
        notifier = EmailNotifier()
        arguments = vars(args)
        notifier.notify(run, arguments, additional_body=str(arguments))
    else:
        run(vars(args))
