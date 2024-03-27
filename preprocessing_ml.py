from data_preprocessing.data_preprocessing_ml import run
import os
from src.loader.paths import CONFIG_DIR


if __name__ == '__main__':

    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f'Directory created at \'{CONFIG_DIR}\'')

    # Adult
    run(dataset_name='phishing', as_category_columns=range(31))

