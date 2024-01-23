from recs import run as run_recs

dataset_name = 'gift'
dataset_type = 'train'

for base_seed in range(1000, 5001, 100):
    args = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'base_seed': base_seed
    }
    run_recs(args)
