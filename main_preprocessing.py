from data_preprocessing.data_preprocessing import run


if __name__ == '__main__':
    run(dataset_name='facebook_books', core=5)
    run(dataset_name='yahoo_movies', core=10)
