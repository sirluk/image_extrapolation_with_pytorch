from pathlib import Path
from PIL import Image
import numpy as np
import dill as pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def get_stats(filepath):
    ar = np.asarray(Image.open(filepath), dtype=np.float32)
    ar /= 255
    return np.array([ar.sum(), np.square(ar).sum(), ar.size])    


print("Enter image directory:")
input_dir = input()

n_cpu = cpu_count()
res = Parallel(n_jobs=max(n_cpu-1, 1))(delayed(get_stats)(f) for f in Path(input_dir).rglob("*.jpg"))
stats = np.stack(res).sum(0)
    
mean = stats[0] / stats[2]
std = np.sqrt(stats[1] / stats[2] - mean**2)

with open("data/image_stats.pkl", "wb") as f:
    pickle.dump([float(mean), float(std)], f)
