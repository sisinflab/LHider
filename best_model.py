import json
import os
import argparse
from best_template import TEMPLATE
from elliot.run import run_experiment

config_dir = 'config_files/'
RANDOM_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
args = parser.parse_args()
dataset_name = args.dataset
metrics = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
           "SEntropy", "EFD", "EPC",  "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]


best_model_path = os.path.join("results", dataset_name, "performance")

for filename in [file for file in os.listdir(best_model_path) if file.endswith('.json')]:

    best_model_file = os.path.join(best_model_path, filename)
    with open(best_model_file, "r") as json_file:
        best_model = json.load(json_file)
        cutoffs = best_model[0]['default_validation_cutoff']
        neighbors = best_model[3]['configuration']['neighbors']
        l2 = best_model[4]['configuration']['l2_norm']
        neighborhood = best_model[5]['configuration']['neighborhood']
        alpha = best_model[5]['configuration']['alpha']
        beta = best_model[5]['configuration']['beta']
        normalize_similarity = best_model[5]['configuration']['normalize_similarity']

    config = TEMPLATE.format(dataset=dataset_name, cutoffs=cutoffs, metrics=metrics, neighbors=neighbors, l2=l2,
                             neighborhood=neighborhood, alpha=alpha, beta=beta, normalize_similarity=normalize_similarity)
    config_path = os.path.join(config_dir, 'best_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)

    run_experiment(config_path)
