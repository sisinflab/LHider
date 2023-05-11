from src.loader.paths import *
import os
import glob
import pandas as pd
import re

GLOBAL_SEED = 42


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the paramenters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    # loading files
    dataset_name = args['dataset']
    output_name = args['output']

    # merge metrics in directory
    merge_metrics(dataset_name=dataset_name, output_name=output_name)


def merge_metrics(dataset_name: str, output_name: str = None):
    files_path = os.path.join(METRIC_DIR, dataset_name + "*", "*.tsv")

    if output_name:
        output_path = os.path.join(METRIC_DIR, f'{output_name}.tsv')
    else:
        output_path = os.path.join(METRIC_DIR, f"{dataset_name}_merged.tsv")

    print('files loading and merging')
    for filename in glob.glob(files_path):
        eps_rr = (re.findall('epsrr([0-9]*)', filename)[0]) if "epsrr" in str(filename) else 0
        eps_exp = (re.findall('epsexp([0-9]*)', filename)[0]) if "epsexp" in str(filename) else 0
        print(f'Reading: \'{filename}\'')
        df = pd.read_csv(filename, sep='\t')
        df.insert(0, 'eps_rr', eps_rr)
        df.insert(1, 'eps_exp', eps_exp)
        df.to_csv(output_path, mode="a+", sep="\t", index=False, header=not os.path.exists(output_path))

    print(f'merged metrics written at \'{output_path}\'')
