# Enhancing Utility in Differentially Private Recommendation Data Release with Exponential Mechanism
This is the official repository for the paper *Enhancing Utility in Differentially Private Recommendation Data Release with Exponential Mechanism* currently under review at SIGIR 2024.


The recommenders' training and evaluation procedures have been developed on the reproducibility framework **Elliot**,
we suggest to refer to the official Github [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

## Table of Contents

- [Requirements](#requirements)
  - [Installation guidelines](#installation-guidelines)
- [Datasests](#datasets)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Generate Datasets with Randomized Response](#generate-datasets-with-randomized-response)
  - [Select Dataset with Exponential Mechanism](#select-dataset-with-exponential-mechanism)
- [Baseline](#baseline)
  - [Recommendation Baseline](#recommendation-baseline)
  - [Subsample Exponential Mechanism](#subsample-exponential-mechanism)
  - [Basic One-Time RAPPOR](#basic-one-time-rappor)

## Requirements

This software has been executed on the operative system Ubuntu `20.04`.

Please have at least Python `3.8.0` installed on your system.

### Installation guidelines

You can create the virtual environment with the requirements files included in the repository, as follows:

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -upgrade pip
pip install -r requirements.txt
```

## Datasets

At `data/`, you may find all the [files](https://github.com/sisinflab/LHider/tree/main/data) related to the datasets. Each dataset can be found in `data/[DATASET_NAME]/data/dataset.tsv`

The datasets used in the paper are `Amazon Gift Card`, `Facebook Books` and `Yahoo! Movies` referred as
`gift`, `facebook_books`, and `yahoo_movies`, respectively. 
## Elliot Configuration Templates

At `config_templates/`, you may find the Elliot [configuration templates](https://github.com/sisinflab/LHider/tree/main/config_templates) used for setting the experiments. 

The configuration template used for all the experiments is `training.py`.

## Usage

Here, we describe the steps to reproduce the results presented in the paper. 

### Preprocessing

Run the data preprocessing step with the following:

```bash
python preprocessing.py
```

This step binarize all the datasets and splits them into train and test sets. The results will be stored in `data/[DATASET_NAME]` for each dataset.

### Generate Datasets with Randomized Response

From the binarized datasets, 500 randomized versions have been generated with the following:

```bash
python generation.py --dataset [DATASET_NAME]
```
The perturbed dataset will be stored in the directory `perturbed_dataset/[DATASET_NAME]_train/0`.

For example, if you want to run the script on the Amazon Gift Card dataset
```bash
python generation.py --dataset gift
```

Each perturbed dataset will be then split in train and validation set, which will be stored in `data/[DATASET_NAME]/generated_train/0`.

Finally, the recommendation performance for each dataset will be stored in `result_collection/[DATASET_NAME]_train/0/`.

### Select Dataset with Exponential Mechanism

We can run the selection module with the following:

```bash
python selection.py --dataset [DATASET_NAME]
```
where [DATASET_NAME] is the name of the dataset.

The results for each model and dataset will be stored in `result_data/[DATASET_NAME]_train/0/[DATASET_NAME]_train_[MODEL_NAME]_nDCGRendle2020.tsv`.

## Baseline

Here we describe the steps to reproduce the baseline presented in the paper. 

### Recommendation Baseline

To reproduce the recommendation performance for the original datasets, run:

```bash
python baseline.py --dataset [DATASET_NAME]
```
where [DATASET_NAME] is the name of the dataset.

The result will be stored in `data/[DATASET_NAME]/baseline`.

### Subsample Exponential Mechanism

Run Subsample Exponential Mechanism with:
```bash
python subsample.py --dataset [DATASET_NAME]
```
where [DATASET_NAME] is the name of the dataset.


The result will be stored in `results_data/[DATASET_NAME]_train/0/aggregated_results.tsv`.

### Basic One-Time RAPPOR

To run One-Time RAPPOR, refer to [Generate Datasets with Randomized Response](#generate-datasets-with-randomized-response).
