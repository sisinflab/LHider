from transform_dataset import run

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'raw'

args = {
    'dataset_name': dataset_name,
    'dataset_type': dataset_type
}

run(args)