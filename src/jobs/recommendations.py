from . import experiment_info
from config_templates.training import TEMPLATE_PATH
from src.loader.paths import *
from elliot.run import run_experiment


def run(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    dataset_name = args['dataset']
    dataset_type = args['type']

    # compute recommendations on the original dataset
    if args['original']:
        dataset_path = dataset_filepath(dataset_name, dataset_type)
    # compute recommendations on the synthetic dataset
    else:
        assert 'eps_rr' in args, 'eps_rr parameter missing'
        assert 'eps_exp' in args, 'eps_exp parameter missing'
        eps_rr = args['eps_rr']
        eps_exp = args['eps_exp']
        score_type = args['score_type']
        dataset_path = synthetic_dataset_filepath(dataset_name, dataset_type, score_type, eps_rr, eps_exp)
        dataset_name = synthetic_dataset_name(dataset_name, dataset_type, score_type, eps_rr, eps_exp)

    # edit the config file
    config = TEMPLATE_PATH.format(dataset=dataset_name, path=dataset_path)
    # write the config file
    config_path = os.path.join(CONFIG_DIR, 'runtime_conf.yml')

    with open(config_path, 'w') as file:
        file.write(config)

    run_experiment(config_path)
