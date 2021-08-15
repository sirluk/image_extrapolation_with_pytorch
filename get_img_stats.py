# Code used to get mean and standard deviation from dataset and save it to file
# Process is parallelized

from pathlib import Path
from PIL import Image
import numpy as np
import dill as pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from src.utils import get_stats


input_dir = "data/data_train"
outfile = "files/image_stats.pkl"

n_cpu = cpu_count()
res = Parallel(n_jobs=max(n_cpu-1, 1))(delayed(get_stats)(f) for f in Path(input_dir).rglob("*.jpg"))
stats = np.stack(res).sum(0)
    
mean = stats[0] / stats[2]
std = np.sqrt(stats[1] / stats[2] - mean**2)

with open(outfile, "wb") as f:
    pickle.dump([float(mean), float(std)], f)
