import argparse
from src.jobs.aggregate import run


def read_arguments(args: argparse.Namespace):
    arguments = ['dataset']
    return {arg: args.__getattribute__(arg) for arg in arguments}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)

arguments = read_arguments(parser.parse_args())
run(arguments)
