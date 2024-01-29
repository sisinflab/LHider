import os
from data_preprocessing.data_preprocessing import run
import pandas as pd

if __name__ == '__main__':
    # Facebook Books
    run(dataset_name='facebook_books', core=5)

    # Yahoo! Movies
    run(dataset_name='yahoo_movies', core=10, threshold=3)

    # Gift Cards
    data_path = os.path.join('data', 'gift', 'data', 'dataset.tsv')
    data = pd.read_csv(data_path, sep=',', header=None, engine='python', names=['i', 'u', 'r', 't'])
    output_folder = os.path.join('data', 'gift', 'data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'dataset.tsv')
    data = data[['u', 'i', 'r']]
    data.to_csv(output_path, sep='\t', header=False, index=False)
    run(dataset_name='gift', core=5, threshold=3)
