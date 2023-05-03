import os

PROJECT_PATH = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
RESULT_DIR = os.path.join(PROJECT_PATH, 'results')
MAIN_DIR = [RESULT_DIR]


def check_main_directories():
    """
    Check that the main directories exist, otherwise create the missing
    @return: None
    """
    for path in MAIN_DIR:
        if not os.path.exists(path):
            os.makedirs(path)


def create_directory(dir_path: str):
    """
    Check that the directory exists, otherwise create it
    @return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def dataset_directory(dataset_name: str):
    """
    Given the dataset name returns the dataset directory
    @param dataset_name: name of the dataset
    @return: the path of the directory containing the dataset data
    """
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f'Directory at {dataset_dir} not found. Please, check that dataset directory exists')
    return os.path.abspath(dataset_dir)


def scores_directory(dataset_dir: str, score: float):
    score_string = str(score)
    scores_dir = os.path.join(dataset_dir, 'scores', 'eps_'+score_string)
    if not os.path.exists(scores_dir):
        raise FileNotFoundError(f'Scores at {scores_dir} not found. Please, check that scores directory exists')
    return os.path.abspath(scores_dir)

