from src.jobs.sigir import run

# definisci i parametri necessari
randomizer = 'randomized'
eps_phi: float = 1.0
reps = 5
eps_exp: float = 1.0
exp_score = 'manhattan'
total_eps = eps_phi * reps + eps_exp

print('eps_phi', eps_phi,
      'reps', reps,
      'eps_exp', eps_exp,
      'total eps', total_eps)

# riproducibilità
base_seed = 100
seed = base_seed

# dataset
dataset_name = 'facebook_books'
dataset_type = 'train'

for eps_phi in [100]:
    for reps in [1]:
        for eps_exp in [100]:
            seed += 1
            # run
            args = {
                'dataset': dataset_name,
                'type': dataset_type,
                'eps_phi': eps_phi,
                'randomizer': randomizer,
                'reps': reps,
                'eps_exp': eps_exp,
                'seed': seed,
                'base_seed': 100,
                'total_eps': eps_phi * reps + eps_exp
            }
            run(args)
