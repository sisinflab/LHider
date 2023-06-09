import argparse
from src.jobs.aggregate import run


# def read_arguments(args: argparse.Namespace):
#     arguments = ['dataset', 'eps']
#     return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--eps', required=True, type=float, nargs='+')

arguments = vars(parser.parse_args())
run(arguments)
