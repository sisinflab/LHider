import json
from elliot.run import run_experiment
from src.loader.paths import *
from config_templates.best import TEMPLATE


def run(dataset_name: str, dataset_file: str, best_json: str, metrics):
    if not os.path.exists(best_json):
        FileNotFoundError(f'File not found at {best_json}. Please, check your files.')

    cutoffs, neighbors, l2, neighborhood, alpha, beta, normalize_similarity = \
        None, None, None, None, None, None, None

    with open(best_json, "r") as json_file:
        best_model = json.load(json_file)

        for obj in best_model:
            cutoffs = obj.get('default_validation_cutoff', cutoffs)
            conf = obj.get('configuration', {})
            neighbors = conf.get('neighbors', neighbors)
            l2 = conf.get('l2_norm', l2)
            neighborhood = conf.get('neighborhood', neighborhood)
            alpha = conf.get('alpha', alpha)
            beta = conf.get('beta', beta)
            normalize_similarity = conf.get('normalize_similarity', normalize_similarity)

    # writing best configuration files
    config = TEMPLATE.format(dataset=dataset_name, file=dataset_file, cutoffs=cutoffs, metrics=metrics,
                             neighbors=neighbors, l2=l2, neighborhood=neighborhood, alpha=alpha, beta=beta,
                             normalize_similarity=normalize_similarity)

    config_path = os.path.join(CONFIG_DIR, 'best_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)

    metrics_path = os.path.join(os.getcwd(), "metrics", dataset_file)
    os.makedirs(metrics_path, exist_ok=True)

    run_experiment(config_path)
