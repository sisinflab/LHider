import pandas as pd
import math
from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import DPCrsMatrix
from src.exponential_mechanism.mechanism import ExponentialMechanism
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
from src.randomize_response.mechanism import RandomizeResponse

GLOBAL_SEED = 42


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']
    dataset_dir = os.path.join(DATA_DIR, dataset_name)

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type="csr")
    data = DPCrsMatrix(loader.load(), path=dataset_path, data_name=dataset_name)

    # dataset result directory
    dataset_result_dir = dataset_directory(dataset_name=dataset_name)
    create_directory(dataset_result_dir)

    # print dataset info
    data.info()

    # privacy budgets
    eps_rr = float(args['eps_rr'])
    esp_exp = [float(e) for e in args['eps_exp']]

    # path of the folder containing the scores
    scores_dir = scores_directory(dataset_dir=dataset_dir, eps=eps_rr, type=dataset_type)
    scores_path = scores_file_path(scores_dir=scores_dir)

    # transform privacy budget value in a feedback change probability for the LHider mechanism
    change_prob = 1 / (1 + math.exp(eps_rr))

    for eps_ in esp_exp:
        c_seed, c_score = exponential_mechanism(scores_path=scores_path, eps=eps_)
        print(f'Selected a dataset with score {c_score} and seed {c_seed}')
        gen_dataset = generate(data=data, change_probability=change_prob, seed=c_seed - GLOBAL_SEED)
        gen_dataframe = pd.DataFrame(zip(gen_dataset.nonzero()[0], gen_dataset.nonzero()[1]))
        result_path = generated_result_path(data_name=data.name, type=dataset_type, eps_rr=eps_rr, eps_exp=eps_)
        gen_dataframe.to_csv(result_path, sep='\t', header=False, index=False)
        print(f'Dataset stored at: \'{result_path}\'')


def generate(data, change_probability: float, seed: int):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    return ratings_generator.privatize(data.values)


def exponential_mechanism(scores_path: str, eps: float):
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f'Scores not found at \'{scores_path}\'')

    score_function = LoadScores(scores_path, sensitivity=1)
    possible_output_seeds = list(score_function.data.keys())
    mech = ExponentialMechanism(score_function, eps)
    chosen_seed = mech.privatize(possible_output_seeds)
    chosen_score = score_function.score_function(chosen_seed)
    return chosen_seed, chosen_score



