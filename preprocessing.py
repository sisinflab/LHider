from data_preprocessing.data_preprocessing import run
import os
from src.loader.paths import CONFIG_DIR


if __name__ == '__main__':

    # create configuration files directory
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f'Directory created at \'{CONFIG_DIR}\'')

    # Facebook Books
    run(dataset_name='facebook_books', core=5)

    # Yahoo! Movies
    # run(dataset_name='yahoo_movies', core=10, threshold=3)

    # Gift Cards
    run(dataset_name='gift', core=5, threshold=3)
