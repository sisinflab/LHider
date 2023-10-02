import os

PROJECT_PATH = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
RESULT_DIR = os.path.join(PROJECT_PATH, 'results')
GENERATED_DIR = os.path.join(PROJECT_PATH, 'generated')
METRIC_DIR = os.path.join(PROJECT_PATH, 'metrics')
CONFIG_DIR = os.path.join(PROJECT_PATH, 'config_files')
RAW_DATA_FOLDER = 'data'
DATASET_NAME = 'dataset.tsv'
DATASET_NAME_BY_TYPE = {
    'raw': os.path.join('data', 'dataset.tsv'),
    'clean': 'dataset.tsv',
    'train': 'train.tsv',
    'val': 'val.tsv',
    'test': 'test.tsv'
}
MAIN_DIR = [RESULT_DIR]


def check_main_directories():
    """
    Check that the main directories exist, otherwise create the missing
    @return: None
    """
    for path in MAIN_DIR:
        create_directory(path)


def create_directory(dir_path: str):
    """
    Check that the directory exists, otherwise create it
    @param dir_path: path of the directory to create
    @return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created directory at \'{dir_path}\'')


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


def dataset_filepath(dataset_name: str, type='raw'):
    """
    Given the dataset name returns the path of the dataset file
    @param dataset_name: name of the dataset
    @param type: type of dataset. Raw, clean, training, validation or test
    @return: the path of the directory containing the dataset data
    """
    assert type in DATASET_NAME_BY_TYPE.keys(), f'Incorrect dataset type. Dataset type found {type}.'
    dataset_dir = dataset_directory(dataset_name)
    filepath = os.path.join(dataset_dir, DATASET_NAME_BY_TYPE[type])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File at {filepath} not found. Please, check your files')
    return os.path.abspath(filepath)


def synthetic_dataset_name(dataset_name: str, type: str, eps_rr: float, eps_exp: float):
    return f'{dataset_name}_{type}_epsrr{eps_rr}_epsexp{eps_exp}'


def synthetic_dataset_filepath(dataset_name: str, type: str, eps_rr: float, eps_exp: float):
    dataset_dir = dataset_directory(dataset_name)
    generated_dir = os.path.join(dataset_dir, 'generated')
    dataset_name = synthetic_dataset_name(dataset_name, type, eps_rr, eps_exp)
    dataset_path = os.path.join(generated_dir, dataset_name) + '.tsv'
    assert os.path.exists(dataset_path), \
        f'dataset at \'{dataset_path}\' not found'
    return dataset_path


def scores_directory(dataset_dir: str, eps: float, score_type: str, type: str = 'clean'):
    """
    Given the dataset directory and the value of epsilon return the score directory
    @param dataset_dir: dataset directory path
    @param eps: value of epsilon
    @param score_type: type of score
    @param type: type of dataset
    @return: the path of the directory containing the scores
    """
    eps_string = str(eps)
    scores_dir = os.path.join(dataset_dir, f'scores_{type}', score_type, 'eps_' + eps_string)
    if not os.path.exists(scores_dir):
        raise FileNotFoundError(f'Scores at {scores_dir} not found. Please, check that scores directory exists')
    return os.path.abspath(scores_dir)


def scores_file_path(scores_dir: str):
    """
    Given a scores directory searches for the scores file path
    @param scores_dir: directory containing the scores file
    @return: path of the scores file
    """
    files_in_dir = os.listdir(scores_dir)
    assert len(files_in_dir) == 1, 'More than one file found in the score directory. ' \
                                   f'Please, check the directory {files_in_dir}'
    path = os.path.abspath(os.path.join(scores_dir, files_in_dir[0]))
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found at {path}. Please, check your files.')
    return path


def dataset_result_directory(dataset_name: str, type: str) -> str:
    """
    @param dataset_name: name of the dataset
    @param type: type of dataset
    @return: path of the directory containing the results for the specific dataset
    """
    assert type in DATASET_NAME_BY_TYPE.keys(), f'Incorrect dataset type. Dataset type found {type}.'
    dataset_result_dir = os.path.join(RESULT_DIR, dataset_name + '_' + type)
    return dataset_result_dir


def generated_result_path(data_name: str, eps_rr: float, eps_exp: float, type: str, score_type: str):
    result_folder = os.path.join(DATA_DIR, data_name, 'generated')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    return os.path.join(result_folder, f'{data_name}_{type}_{score_type}_epsrr{str(eps_rr)}_epsexp{str(eps_exp)}.tsv')


def score_directory(dataset_name, eps_rr, dataset_type, score_type):
    """
    Returns the path of the scores folder
    @param dataset_name: name of the dataset and name of the folder
    @param eps_rr: epsilon value for randomized response
    @param dataset_type: type of dataset
    @param score_type: type of score (manhattan, euclidean, ...)
    @return: path of the directory containing the scores
    """
    eps = f'eps_{eps_rr}'
    folder_path = os.path.abspath(os.path.join(DATA_DIR, dataset_name, 'scores', dataset_type, score_type, eps))
    return folder_path


def create_score_directory(dataset_name, eps_rr, dataset_type, score_type):
    """
    Create the folder where the scores will be stored
    @param dataset_name: name of the dataset and name of the folder
    @param eps_rr: epsilon value for randomized response
    @param dataset_type: type of dataset (clean, raw, train)
    @param score_type: type of score (manhattan, euclidean, ...)
    @return: path of the directory containing the scores
    """
    folder_path = score_directory(dataset_name, eps_rr, dataset_type, score_type)
    create_directory(folder_path)
    return folder_path
