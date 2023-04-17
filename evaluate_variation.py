import os
import argparse
import numpy as np
from src.create_dataset import user_item_rating_matrix_from_dataset


def evaluate_total_variation(original_matrix, result_matrix):
    different = sum(sum(original_matrix != result_matrix))
    size = original_matrix.size
    return different / size * 100


def evaluate_ones_variation(original_matrix, result_matrix):
    equal = sum(sum(np.logical_and(original_matrix, result_matrix)))
    original_ones_number = sum(sum(original_matrix))
    result_ones_number = sum(sum(result_matrix))
    ones_variation = (1 - equal / original_ones_number) * 100

    return original_ones_number, result_ones_number, ones_variation


parser = argparse.ArgumentParser()
parser.add_argument('--original_dataset', required=True)
parser.add_argument('--result_root_directory', required=False, default='results')

args = parser.parse_args()
dataset_path = args.original_dataset
dataset_name = os.path.split(dataset_path)[-1].split('.')[0]
result_directory = args.result_root_directory

rating_matrix = user_item_rating_matrix_from_dataset(dataset_path)

print(f'dataset: {dataset_name}')

for root, directory, files in os.walk(result_directory):
    for filename in sorted(files):
        file_path = os.path.join(root, filename)
        eps = filename.split("_")[2]
        eps = ".".join([eps[0], eps[1:]]) if eps[0] == '0' else eps
        result = user_item_rating_matrix_from_dataset(file_path)
        variation_percentage = evaluate_total_variation(rating_matrix, result)
        original_ones_number, result_ones_number, ones_variation = evaluate_ones_variation(rating_matrix, result)

        print(f'epsilon: {eps} '
              f'\nnumero di 1 originale: {original_ones_number}'
              f'\nnumero di 1 risultato: {original_ones_number}'
              f'\nvariazione sugli 1: {ones_variation}%'
              f'\nvariazione totale: {variation_percentage}%')
