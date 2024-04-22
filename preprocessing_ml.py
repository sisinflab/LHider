from data_preprocessing.data_preprocessing_ml import run
import os
from src.loader.paths import CONFIG_DIR

if __name__ == '__main__':

    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f'Directory created at \'{CONFIG_DIR}\'')

    # Adult
    run(dataset_name='adult', sep='\t', drop_na_value=True, categorical_to_one_hot=True,
        numeric_to_categorical_columns=[0, 2, 4, 10, 11, 12], as_category_columns=[1, 3, 5, 6, 7, 8, 9, 13, 14],
        remove_last_column=True)

    # Phishing
    run(dataset_name='phishing', sep=',', categorical_to_one_hot=True, as_category_columns=range(31),
        remove_last_column=True)
