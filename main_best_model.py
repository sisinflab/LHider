import glob
import argparse
from best_model import run
from src.loader.paths import *

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]


def dataset_from_performance_file(file: str):
    """
    Given the path of the json file containing the best model parameters returns the dataset name
    @param file: path of the best configuration json file
    @return: string containing the name of the dataset related to the best model parameters file
    """
    return os.path.normpath(file).split(os.sep)[-3]


def find_best_performance_files(dataset, dataset_name: str):
    path_format_string = os.path.join(RESULT_DIR,
                                      f'{dataset_name}',
                                      'performance',
                                      'bestmodelparams_*.json')
    json_performance_files = glob.glob(path_format_string)
    return [{'dataset_name': dataset, 'best_json': j} for j in json_performance_files]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument('--type', choices=['raw', 'clean', 'train', 'val', 'test'], default='clean')
parser.add_argument('--eps_rr', required=True, type=float, help='privacy budget of generated datasets')
parser.add_argument('--eps_exp', required=False, type=float, help='exponential mechanism privacy budget')
parser.add_argument('--seed', required=False, type=int, default=42, help='random seed')
parser.add_argument("--metrics", type=str, nargs='+', default=DEFAULT_METRICS)

args = parser.parse_args()

dataset_name = f'{args.dataset}_{args.type}_epsrr{args.eps_rr}_epsexp{args.eps_exp}'

for arg in find_best_performance_files(dataset=args.dataset, dataset_name=dataset_name):
    run(**arg, dataset_file=dataset_name, metrics=args.metrics)
