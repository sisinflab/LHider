import glob
import json
import argparse
from config_templates.best import TEMPLATE
from elliot.run import run_experiment
from src.loader.paths import *

config_dir = 'config_files/'
RANDOM_SEED = 42
metrics = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
           "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]


def dataset_from_performance_file(file: str):
    """
    Given the dataset name and the path of the json file containing the best model parameters returns the specif
    dataset name
    @param dataset_name: base name of the dataset
    @param file: path of the best configuration json file
    @return: string containing the name of the dataset related to the best model parameters file
    """
    #dataset = [token for token in os.path.normpath(file).split(os.sep) if token.startswith(dataset_name)][0]
    return os.path.normpath(file).split(os.sep)[-3]


def find_best_performance_files(dataset_name: str):
    path_format_string = os.path.join(RESULT_DIR,
                                      f'{dataset_name}*',
                                      'performance',
                                      'bestmodelparams_*.json')
    json_performance_files = glob.glob(path_format_string)
    return [{'dataset_name': dataset_from_performance_file(file=j), 'best_json': j}
            for j in json_performance_files]


def run_best(dataset_name: str, best_json: str):

    if not os.path.exists(best_json):
        FileNotFoundError(f'File not found at {best_json}. Please, check your files.')

    with open(best_json, "r") as json_file:
        best_model = json.load(json_file)
        cutoffs = best_model[0]['default_validation_cutoff']
        neighbors = best_model[3]['configuration']['neighbors']
        l2 = best_model[4]['configuration']['l2_norm']
        neighborhood = best_model[5]['configuration']['neighborhood']
        alpha = best_model[5]['configuration']['alpha']
        beta = best_model[5]['configuration']['beta']
        normalize_similarity = best_model[5]['configuration']['normalize_similarity']

    config = TEMPLATE.format(dataset=dataset_name, cutoffs=cutoffs, metrics=metrics, neighbors=neighbors, l2=l2,
                             neighborhood=neighborhood, alpha=alpha, beta=beta,
                             normalize_similarity=normalize_similarity)

    config_path = os.path.join(config_dir, 'best_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)

    metrics_path = os.path.join(os.getcwd(), "metrics", dataset_name)
    os.makedirs(metrics_path, exist_ok=True)

    run_experiment(config_path)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
#parser.add_argument("--json", required=True, type=str)

args = parser.parse_args()

for arg in find_best_performance_files(dataset_name=args.dataset):
    run_best(**arg)
