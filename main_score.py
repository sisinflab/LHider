import argparse
from src.jobs.score import run
from src.jobs.score import SCORERS
import multiprocessing as mp

possible_scores = list(SCORERS.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--type', choices=['raw', 'clean', 'train'], default='clean')
    parser.add_argument('--score_type', choices=possible_scores, default='manhattan')
    parser.add_argument('--eps', required=False, type=float, default=1)
    parser.add_argument('--base_seed', required=False, type=int, default=42)
    parser.add_argument('--start', required=False, type=int, default=0)
    parser.add_argument('--end', required=False, type=int, default=100)
    parser.add_argument('--batch', required=False, type=int, default=10)
    parser.add_argument('--proc', required=False, default=mp.cpu_count()-1, type=int)
    parser.add_argument('--mail', action='store_true')
    args = parser.parse_args()

    if args.mail:
        from email_notifier.email_sender import EmailNotifier
        notifier = EmailNotifier()
        arguments = vars(args)
        notifier.notify(run, arguments, additional_body=str(arguments))
    else:
        run(vars(args))
