from src.jobs.sigir_itemknn import run_generation
from split_generated import run as run_split
from recs import run as run_recs

# definisci i parametri necessari
randomizer = 'randomized'
exp_score = 'jaccard'

# dataset
dataset_name = 'facebook_books'
dataset_type = 'train'
n = 170
folder = 1
seed = 171

def fun():
    for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        # run
        args = {
            'dataset': dataset_name,
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'type': dataset_type,
            'eps_phi': eph_phi,
            'randomizer': randomizer,
            'base_seed': folder,
            'score_type': exp_score,
            'generations': n,
            'seed': seed
        }
        run_generation(args)
    run_split(args)
    run_recs(args)

from email_notifier.email_sender import EmailNotifier
notifier = EmailNotifier()
arguments = {'Esperimento': 'ultima speranza facebook_books 170 340'}
notifier.notify(fun, additional_body=str(arguments))
