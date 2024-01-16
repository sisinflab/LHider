from split_generated import run as run_split
from recs import run as run_recs


def fun(*args, **kwargs):
    dataset_name = 'yahoo_movies'
    dataset_type = 'train'

    for base_seed in range(100, 1001, 100):
        args = {
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'base_seed': base_seed
        }
        run_split(args)

    for base_seed in range(100, 1001, 100):
        args = {
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'base_seed': base_seed
        }
        run_recs(args)


from email_notifier.email_sender import EmailNotifier
notifier = EmailNotifier()
arguments = {'Esperimento': 'raccomandazione'}
notifier.notify(fun, additional_body=str(arguments))
