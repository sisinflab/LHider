import json
import glob
from src.jobs import experiment_info
from elliot.run import run_experiment
from src.loader.paths import *
from config_templates.best import TEMPLATE_PATH


def find_best_performance_files(dataset_name: str) -> str:
    path_format_string = os.path.join(RESULT_DIR, f'{dataset_name}', 'performance', 'bestmodelparams_*.json')
    json_performance_files = glob.glob(path_format_string)

    if not json_performance_files:
        raise FileNotFoundError(f'No model parameters file found. Please, check your files.')

    return max(json_performance_files, key=os.path.getctime)


def run(args: dict):

    # print information about the experiment
    experiment_info(args)

    dataset_name = args['dataset']
    dataset_type = args['type']

    metrics = args['metrics']

    # compute recommendations on the original dataset
    if args['original']:
        dataset_path = dataset_filepath(dataset_name)
    # compute recommendations on the synthetic dataset
    else:
        assert 'eps_rr' in args, 'eps_rr parameter missing'
        assert 'eps_exp' in args, 'eps_exp parameter missing'
        eps_rr = args['eps_rr']
        eps_exp = args['eps_exp']
        score_type = args['score_type']
        dataset_path = synthetic_dataset_filepath(dataset_name, dataset_type, score_type, eps_rr, eps_exp)
        dataset_name = synthetic_dataset_name(dataset_name, dataset_type, score_type, eps_rr, eps_exp)

    best_json = find_best_performance_files(dataset_name=dataset_name)

    if not os.path.exists(best_json):
        raise FileNotFoundError(f'File not found at {best_json}. Please, check your files.')

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

    # path of the output folder
    metrics_path = os.path.join(METRIC_DIR, dataset_name)

    # best configuration file
    config = TEMPLATE_PATH.format(dataset=dataset_name, output_path=metrics_path, path=dataset_path, cutoffs=cutoffs,
                                  metrics=metrics, neighbors=neighbors, l2=l2, neighborhood=neighborhood, alpha=alpha,
                                  beta=beta, normalize_similarity=normalize_similarity)

    config_path = os.path.join(CONFIG_DIR, 'best_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)

    os.makedirs(metrics_path, exist_ok=True)

    run_experiment(config_path)
