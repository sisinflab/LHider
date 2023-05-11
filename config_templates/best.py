TEMPLATE = """experiment:
  dataset: {dataset}
  path_output_rec_performance: ./metrics/{dataset}
  data_config:
    strategy: dataset
    dataset_path: ../data/{dataset}/dataset.tsv
  top_k: 10
  splitting:
    test_splitting:
      strategy: random_subsampling
      test_ratio: 0.2
  evaluation:
    cutoffs: {cutoffs}
    simple_metrics: {metrics}
  gpu: 0
  models:
    Random:
      meta:
        verbose: True
        save_recs: True
      seed: 42
    MostPop:
      meta:
        verbose: True
        save_recs: True
    ItemKNN:
      meta:
        save_recs: True
        verbose: True
        hyper_max_evals: 10
        hyper_opt_alg: tpe
      neighbors: {neighbors}
      similarity: cosine
      implementation: aiolli
      seed: 42
    EASER:
      meta:
        verbose: True
        save_recs: True
        hyper_max_evals: 10
        hyper_opt_alg: tpe
      l2_norm: {l2}
      seed: 42
    RP3beta:
      meta:
        hyper_max_evals: 10
        hyper_opt_alg: tpe
        verbose: True
        save_recs: True
      neighborhood: {neighborhood}
      alpha: {alpha}
      beta: {beta}
      normalize_similarity: {normalize_similarity}
      seed: 42"""
