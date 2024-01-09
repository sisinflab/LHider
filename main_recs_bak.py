import argparse
from utils.recs_best_model import run

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--type', choices=['raw', 'clean', 'train', 'val', 'test'], default='clean')
    parser.add_argument('--score_type', choices=['manhattan', 'euclidean', 'cosineUser', 'cosineItem', 'jaccard'],
                        default='manhattan')
    parser.add_argument('--eps_rr', required=False, type=float, help='privacy budget of generated datasets')
    parser.add_argument('--eps_exp', required=False, type=float, help='exponential mechanism privacy budget')
    parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')
    parser.add_argument("--metrics", type=str, nargs='+', default=DEFAULT_METRICS)
    parser.add_argument('--mail', action='store_true')

    args = parser.parse_args()

    if args.mail:
        from email_notifier.email_sender import EmailNotifier
        notifier = EmailNotifier()
        arguments = vars(args)
        notifier.notify(run, arguments, additional_body=str(arguments))
    else:
        run(vars(args))
