import pandas as pd


def user_item_rating_matrix_from_dataset(dataset_path):
    matrix = pd.read_csv(dataset_path, sep="\t", header=None)
    # user item rating matrix from pd.DataFrame
    matrix = matrix.pivot(index=0, columns=1, values=2)
    matrix = matrix.fillna(0)
    return matrix.to_numpy()


def dataset_from_user_item_rating_matrix(matrix):
    dataset = pd.DataFrame(matrix)
    dataset = pd.melt(dataset.reset_index(), id_vars='index')
    dataset = dataset.rename(columns={'index': 'user', 'variable': 'item', 'value': 'rating'})
    # se rimuovo un intera riga o un intera colonna potrei perdere user/item nella rating matrix
    # dataset = dataset[dataset['rating'] != 0]
    dataset = dataset.sort_values(by=['user', 'item'])
    return dataset
