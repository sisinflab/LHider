from recs import run as run_recs

dataset_name = 'yahoo_movies'
dataset_type = 'train'


def fun():
    for base_seed in range(2200, 3601, 100):
        args = {
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'base_seed': base_seed
        }
        run_recs(args)


from email_notifier.email_sender import EmailNotifier
notifier = EmailNotifier()
arguments = {'Esperimento': f'sola raccomandazione su {dataset_name}'}
notifier.notify(fun, additional_body=str(arguments))
