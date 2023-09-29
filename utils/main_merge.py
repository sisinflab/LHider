import argparse
from src.jobs.merge import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=False)

arguments = vars(parser.parse_args())
run(arguments)
