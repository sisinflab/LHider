from src.jobs.sigir import run

# definisci i parametri necessari
randomizer = 'randomized'
eps_phi: float = 1.0
reps = 5
eps_exp: float = 1.0
exp_score = 'manhattan'
total_eps = eps_phi*reps+eps_exp

print('eps_phi', eps_phi,
      'reps', reps,
      'eps_exp', eps_exp,
      'total eps', total_eps)

# riproducibilità
seed = 42

# dataset
dataset_name = 'facebook_books'
dataset_type = 'raw'


for eps_phi in [1.0, 2.0, 5.0, 10.0]:
      for reps in [1, 2, 5, 10]:
            for eps_exp in [1.0, 2.0, 5.0, 10.0]:
                  # run
                  args = {
                        'dataset': dataset_name,
                        'type': dataset_type,
                        'eps_phi': eps_phi,
                        'randomizer': randomizer,
                        'reps': reps,
                        'eps_exp': eps_exp
                  }
                  run(args)
