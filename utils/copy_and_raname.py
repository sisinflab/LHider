import glob
import os
import shutil
from src.loader.paths import METRIC_DIR

path = os.path.join(METRIC_DIR, "*", "*.tsv")

for f in glob.glob(path):
    name = os.path.dirname(f).split(os.path.sep)[-1]
    f_name = name + ".tsv"
    dest = os.path.join(METRIC_DIR, f_name)
    shutil.copy(f, dest)
