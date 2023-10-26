import pandas as pd
import glob
import os
from src.loader.paths import METRIC_DIR

path = os.path.join(METRIC_DIR, "*.tsv")
directory = os.path.dirname(path)

for f in glob.glob(path):
    df = pd.read_csv(f, sep="\t")
    df['model'] = df['model'].str.split('_').str[0]
    name = os.path.basename(f).split(".tsv")[0] + ".tex"
    f_path = os.path.join(directory, name)
    df.style.hide(axis="index").to_latex(f_path, hrules=True)
