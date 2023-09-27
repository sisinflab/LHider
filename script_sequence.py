import subprocess
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, nargs='+')
    parser.add_argument('--type', choices=['raw', 'clean', 'train', 'val', 'test'], default='clean')
    parser.add_argument('--epsrr', required=True, type=str, nargs='+')
    parser.add_argument('--epsexp', required=True, type=str, nargs='+')
    parser.add_argument('--scripts', required=False, type=str, nargs='+',
                        default=["main_lhider.py", "compute_recommendation.py", "main_best_model.py", "main_merge.py"])

    args = parser.parse_args()

    dataset = args.dataset
    epsrr = args.epsrr
    epsexp = args.epsexp
    type = args.type
    script_list = args.scripts

    dataset_names = [f'{d}_{type}_epsrr{rr}_epsexp{e}' for d in dataset for rr in epsrr for e in epsexp]

    for script in script_list:
        if script == "main_lhider.py":
            for d in dataset:
                for rr in epsrr:
                    command = [sys.executable, "main_lhider.py", "--dataset", d, "--eps_rr", rr, "--type", type, "--eps_exp"] + epsexp
                    subprocess.run(command)
                    print(f"Finished: {script}")
        else:
            for d in dataset:
                for rr in epsrr:
                    for exp in epsexp:
                        if script != "main_merge.py":
                            command = [sys.executable, script, "--dataset", d, "--eps_rr", rr, "--eps_exp", exp, "--type", type]
                        else:
                            command = [sys.executable, script, "--dataset", d]
                        subprocess.run(command)
                        print(f"Finished: {script}")
