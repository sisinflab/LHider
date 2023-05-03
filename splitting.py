from data_preprocessing import dataset_preprocessing

facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'
privatized_folders = ['./data/facebook_book_eps6_1']

if __name__ == '__main__':
    not_privatized_folders = [facebook_book_folder, yahoo_movies_folder]

    folders = not_privatized_folders + privatized_folders

    for folder in folders:
        dataset_preprocessing.run(data_folder=folder)
