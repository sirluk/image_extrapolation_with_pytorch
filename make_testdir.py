# File used to randomly select some images from train set and move them to a separate folder as a testset

from pathlib import Path
from numpy.random import default_rng
import numpy as np


root_dir = "data/data_train"
testset_dir = "data/data_test"

root_dir = Path(root_dir)
file_paths = sorted([str(p) for p in root_dir.rglob("*.jpg")])

Path(testset_dir).mkdir(parents=True, exist_ok=True)

rng = default_rng(seed=0)
numbers = rng.choice(len(file_paths), size=208, replace=False)
testset_paths = [file_paths[idx] for idx in numbers]
for p in testset_paths:
    fname = "_".join(p.split("/")[-3:])
    Path(p).rename(f"{testset_dir}/{fname}.jpg")
