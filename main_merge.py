import argparse
from src.jobs.merge import run


# def read_arguments(args: argparse.Namespace):
#     arguments = ['dataset', 'output']
#     return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=False)

arguments = vars(parser.parse_args())
run(arguments)
