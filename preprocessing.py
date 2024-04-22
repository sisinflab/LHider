from data_preprocessing.data_preprocessing import run
import os
from src.loader.paths import CONFIG_DIR
import pandas as pd


def load_movielens(path):
    return pd.read_csv(path, sep='::', header=None, engine='python', names=['u', 'i', 'r', 't'])

if __name__ == '__main__':

    # create configuration files directory
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f'Directory created at \'{CONFIG_DIR}\'')

    # Facebook Books
    run(dataset_name='facebook_books', core=5)

    # Yahoo! Movies
    run(dataset_name='yahoo_movies', core=10, threshold=3)

    # Gift Cards
    run(dataset_name='gift', core=5, threshold=3)

    dataframe = load_movielens('data/movielens/data/ratings.dat')
    dataframe[['u', 'i', 'r']].to_csv('data/movielens/data/dataset.tsv', sep='\t', index=False, header=False)
    # MovieLens
    run(dataset_name='movielens', core=5, threshold=3)
