import argparse
from src.jobs.aggregate import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--eps', required=True, type=float, nargs='+')

    arguments = vars(parser.parse_args())
    run(arguments)
