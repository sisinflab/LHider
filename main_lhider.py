import argparse
from src.jobs.lhider import run


def read_arguments(args: argparse.Namespace):
    arguments = ['dataset', 'eps_rr', 'eps_exp', 'seed']
    return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='name of the dataset that is going to be anonymized')
parser.add_argument('--eps_rr', required=False, type=float, default=0.1)
parser.add_argument('--eps_exp', required=False, type=float, nargs='+', help='exponential mechanism privacy budget')
parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')

args = parser.parse_args()
run(read_arguments(args))
