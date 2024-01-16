TEMPLATE = """experiment:
  dataset: {file}
  data_config:
    strategy: dataset
    dataset_path: ../data/{dataset}/{file}.tsv
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
    MostPop:
      meta:
        verbose: True
        save_recs: False
    EASER:
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 10
        hyper_opt_alg: tpe
      l2_norm: [uniform, 10, 10e7]
      seed: 42
      validation_metric: nDCGRendle2020@10
"""

TEMPLATE_PATH = """experiment:
  dataset: {dataset}
  path_output_rec_performance: {output_path}
  data_config:
    strategy: fixed
    train_path: {train_path}
    validation_path: {val_path}
    test_path: {test_path}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020, Recall, HR, nDCG, Precision, F1, MAP, MAR, LAUC, ItemCoverage, Gini, SEntropy,EFD, EPC,  PopREO, PopRSP, ACLT, APLT, ARP]
  gpu: 0
  models:
    MostPop:
      meta:
        verbose: True
        save_recs: False
    EASER:
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      l2_norm: [uniform, 10, 10e5]
      seed: 42
    ItemKNN:
      meta:
        save_recs: False
        verbose: True
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      neighbors: [uniform, 5, 30]
      similarity: cosine
      implementation: aiolli
      seed: 42
      validation_metric: nDCGRendle2020@10
"""
