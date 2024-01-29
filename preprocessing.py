import os
from data_preprocessing.data_preprocessing import run
import pandas as pd

if __name__ == '__main__':
    # Facebook Books
    run(dataset_name='facebook_books', core=5)

    # Yahoo! Movies
    run(dataset_name='yahoo_movies', core=10, threshold=3)

    # Gift Cards
    run(dataset_name='gift', core=5, threshold=3)
