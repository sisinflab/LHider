import subprocess
import sys
import argparse
import glob
from src.loader.paths import *


def script_paths(script_names: list):
    scripts = []

    for s in script_names:
        scripts += glob.glob(f'{PROJECT_PATH}/**/{s}', recursive=True)

    return scripts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, nargs='+')
    parser.add_argument('--type', choices=['raw', 'clean', 'train', 'val', 'test'], default=['clean'], nargs='+')
    parser.add_argument('--score_type', required=True,
                        choices=['manhattan', 'euclidean', 'cosineUser', 'cosineItem', 'jaccard'], nargs='+')
    parser.add_argument('--eps_rr', required=True, type=str, nargs='+')
    parser.add_argument('--eps_exp', required=True, type=str, nargs='+')
    parser.add_argument('--scripts', required=False, type=str, nargs='+',
                        default=["main_lhider.py", "main_best.py", "main_recs.py"])

    args = parser.parse_args()

    dataset = args.dataset
    eps_rr = args.eps_rr
    eps_exp = args.eps_exp
    dataset_type = args.type
    score_type = args.score_type
    script_list = script_paths(args.scripts)

    for script in script_list:
        for d in dataset:
            for t in dataset_type:
                for s in score_type:
                    for rr in eps_rr:
                        for exp in eps_exp:
                            command = [sys.executable, script, "--dataset", d, "--eps_rr", rr, "--eps_exp", exp,
                                       "--type", t, "--score_type", s]
                            subprocess.run(command)
                            print(f"Finished: {script}")
