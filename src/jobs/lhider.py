import math

from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import DPCrsMatrix
from src.exponential_mechanism.mechanism import ExponentialMechanism
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
from src.randomize_response.mechanism import RandomizeResponse
import pandas as pd
import os

GLOBAL_SEED = 42


def scores_file(scores_dir: str):
    """
    Given a scores directory searches for the score file
    @param scores_dir: directory containing the scores file
    @return: path of the scores file
    """
    files_in_dir = os.listdir(scores_dir)
    assert len(files_in_dir) == 1, 'More than one file found in the score directory. ' \
                                   f'Please, check the directory {files_in_dir}'
    scores_file_path = os.path.abspath(os.path.join(scores_dir, files_in_dir[0]))
    if not os.path.exists(scores_file_path):
        raise FileNotFoundError(f'File not found at {scores_file_path}. Please, check your files.')
    return scores_file_path


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args):
    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_dir = os.path.join(DATA_DIR, dataset_name)

    # loading files
    dataset_path = os.path.join(dataset_dir, 'dataset.tsv')
    loader = TsvLoader(path=dataset_path, return_type="csr")
    data = DPCrsMatrix(loader.load(), path=dataset_path)

    # dataset result directory
    dataset_result_dir = dataset_directory(dataset_name=dataset_name)
    create_directory(dataset_result_dir)

    # print dataset info
    # TODO: implementare qui il metodo info della classe dataset
    print(f'data ratings: {data.transactions}')
    print(f'data users: {data.n_users}')
    print(f'data items: {data.n_items}')

    # privacy budgets
    eps_rr = float(args['eps_rr'])
    esp_exp = args['eps_exp']

    # path of the folder containing the scores
    scores_dir = scores_directory(dataset_dir=dataset_dir, score=eps_rr)
    scores_path = scores_file(scores_dir=scores_dir)

    # transform privacy budget value in a feedback change probability for the LHider mechanism
    change_prob = 1 / (1 + math.exp(eps_rr))

    for eps_ in esp_exp:
        c_seed, c_score = exponential_mechanism(scores_path=scores_path, eps=eps_)
        print(f'Selected a dataset with score {c_score} and seed {c_seed}')
        gen_dataset = generate(data=data, change_probability=change_prob, seed=c_seed-GLOBAL_SEED)
        gen_dataframe = pd.DataFrame(zip(gen_dataset.nonzero()[0], gen_dataset.nonzero()[1]))
        result_path = os.path.join(f'{RESULT_DIR}', f'{data.name}_p{change_prob}_eps{eps_}.tsv')
        gen_dataframe.to_csv(result_path, sep='\t', header=False, index=False)
        print(f'Dataset stored at: \'{result_path}\'')

def generate(data, change_probability, seed):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    return ratings_generator.privatize(data.values)


def exponential_mechanism(scores_path: str, eps: float):
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f'Scores not found at \'{scores_path}\'')

    score_function = LoadScores(scores_path, sensitivity=1)
    scores = list(score_function.data.keys())
    mech = ExponentialMechanism(score_function, eps)
    chosen_seed = mech.privatize(scores)
    chosen_score = score_function.score_function(chosen_seed)
    return chosen_seed, chosen_score

