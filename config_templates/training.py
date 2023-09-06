TEMPLATE = """experiment:
  dataset: {file}
  data_config:
    strategy: dataset
    dataset_path: ../data/{dataset}/generated/{file}.tsv
  top_k: 10
  splitting:
    test_splitting:
      strategy: random_subsampling
      test_ratio: 0.2
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  models:
    Random:
      meta:
        verbose: True
        save_recs: False
      seed: 42
    MostPop:
      meta:
        verbose: True
        save_recs: False
    ItemKNN:
      meta:
        save_recs: False
        verbose: True
        hyper_max_evals: 10
        hyper_opt_alg: tpe
      neighbors: [uniform, 5, 30]
      similarity: cosine
      implementation: aiolli
      seed: 42
      validation_metric: nDCGRendle2020@10
    EASER:
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 10
        hyper_opt_alg: tpe
      l2_norm: [uniform, 10, 10e7]
      seed: 42
      validation_metric: nDCGRendle2020@10
    RP3beta:
      meta:
        hyper_max_evals: 10
        hyper_opt_alg: tpe
        verbose: True
        save_recs: False
      neighborhood: [uniform, 5, 1000]
      alpha: [uniform, 0, 2]
      beta: [uniform, 0, 2]
      normalize_similarity: [True, False]
      seed: 42
      validation_metric: nDCGRendle2020@10
"""
