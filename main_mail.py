import argparse
from src.run import run, run_mp
import multiprocessing as mp
from email_notifier import email_sender


def read_arguments(args: argparse.Namespace):
    arguments = ['dataset', 'change_prob', 'exp_eps', 'seed',
                 'randomize_seed', 'start', 'end', 'batch', 'job', 'proc']
    return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--change_prob', required=False, type=float, default=0.002)
parser.add_argument('--exp_eps', required=False, type=float, nargs='+')
parser.add_argument('--seed', required=False, type=int, default=42)
parser.add_argument('--randomize_seed', required=False, type=int, default=42)
parser.add_argument('--start', required=False, type=int, default=0)
parser.add_argument('--end', required=False, type=int, default=100)
parser.add_argument('--batch', required=False, type=int, default=10)
parser.add_argument('--job', required=False, default='identity')
parser.add_argument('--proc', required=False, default=mp.cpu_count()-1, type=int)
parser.add_argument('--final_path', required=False, type=str)


args = parser.parse_args()

email_reference = ''
senders, receivers, messages = None, None, None
if email_reference:
    senders = '_'.join([email_reference, 'info.txt'])
    receivers = '_'.join([email_reference, 'info.txt'])
    messages = '_'.join([email_reference, 'message.txt'])

try:
    run(read_arguments(args))
    email_sender.send_email(senders=senders, receivers=receivers, messages=messages)
except Exception as e:
    print(f'An error occurred. Exception {e}')
    print('Sending emails')
    email_sender.send_email(senders=senders, receivers=receivers, messages=messages, error=True, error_exception=e)

