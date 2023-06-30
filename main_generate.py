import argparse
from src.jobs.generate import run
import multiprocessing as mp


# def read_arguments(argparse_arguments: argparse.Namespace):
#     job_arguments = ['dataset', 'eps', 'base_seed', 'start', 'end', 'batch', 'proc']
#     return {arg: argparse_arguments.__getattribute__(arg) for arg in job_arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
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
    arguments = read_arguments(parser.parse_args())
    notifier.notify(run, arguments, additional_body=str(arguments))
else:
    run(read_arguments(args))
