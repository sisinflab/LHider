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
