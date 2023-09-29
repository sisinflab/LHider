import argparse
from src.jobs.analysis import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--change_prob', required=False, type=float, default=0.1)
    parser.add_argument('--seed', required=False, type=int, default=42)
    parser.add_argument('--scores', required=False, type=str)

    args = vars(parser.parse_args())
    run(args)
