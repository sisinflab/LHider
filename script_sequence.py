import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str, nargs='+')
parser.add_argument('--epsrr', required=True, type=float, nargs='+')
parser.add_argument('--epsexp', required=True, type=float, nargs='+')

args = parser.parse_args()

dataset = args.dataset
epsrr = args.epsrr
epsexp = args.epsexp

dataset_names = [f'{d}_epsrr{rr}_epsexp{e}' for d in dataset for rr in epsrr for e in epsexp]
script_list = ["main_lhider.py", "compute_recommendation.py", "main_best_model.py", "main_merge.py"]

epsexp_string = ' '.join([str(e) for e in epsexp])

for script in script_list:
    if script == "main_lhider.py":
        for d in dataset:
            for rr in epsrr:
                command = f'{sys.executable} main_lhider.py --dataset {d} --eps_rr {rr} --eps_exp {epsexp_string}'
                subprocess.run(command)
                print(f"Finished: {script}")
    else:
        for dataset in dataset_names:
            command = f'{sys.executable} {script} --dataset {dataset}'
            subprocess.run(command)
            print(f"Finished: {script}")
