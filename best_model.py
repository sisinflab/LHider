import json
from elliot.run import run_experiment
from src.loader.paths import *
from config_templates.best import TEMPLATE


def run(dataset_name: str, best_json: str, metrics):
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

    config_path = os.path.join(CONFIG_DIR, 'best_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)

    metrics_path = os.path.join(os.getcwd(), "metrics", dataset_name)
    os.makedirs(metrics_path, exist_ok=True)

    run_experiment(config_path)
