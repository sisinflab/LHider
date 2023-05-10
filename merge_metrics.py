import os
import glob
import pandas as pd
import re

dataset_name = "FacebookBooks"
base_path = os.path.join(os.getcwd(), "metrics")
files_path = os.path.join(base_path, dataset_name + "*", "*.tsv")
output_path = os.path.join(base_path, f"{dataset_name}_merged.tsv")

for filename in glob.glob(files_path):
    eps_rr = (re.findall('epsrr([0-9]*)', filename)[0]) if "epsrr" in str(filename) else 0
    eps_exp = (re.findall('epsexp([0-9]*)', filename)[0]) if "epsexp" in str(filename) else 0
    df = pd.read_csv(filename, sep='\t')
    df.insert(0, 'eps_rr', eps_rr)
    df.insert(1, 'eps_exp', eps_exp)
    df.to_csv(output_path, mode="a+", sep="\t", index=False, header=not os.path.exists(output_path))
