import argparse
from src.jobs.analysis import run


# def read_arguments(args: argparse.Namespace):
#     arguments = ['dataset', 'change_prob', 'seed', 'scores']
#     return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--change_prob', required=False, type=float, default=0.1)
parser.add_argument('--seed', required=False, type=int, default=42)
parser.add_argument('--scores', required=False, type=str)

args = vars(parser.parse_args())
run(args)
