from pathlib import Path
from PIL import Image
import numpy as np
import dill as pickle


print("Enter image directory:")
input_dir = input()

running_stats = np.zeros((3,), dtype=np.float32)
for f in Path(input_dir).rglob("*.jpg"):
    ar = np.asarray(Image.open(f), dtype=np.float32)
    ar /= 255
    running_stats += np.array([ar.sum(), np.square(ar).sum(), ar.size])
    
mean = running_stats[0] / running_stats[2]
std = np.sqrt(running_stats[1] / running_stats[2] - mean**2)

with open("data/image_stats.pkl", "wb") as f:
    pickle.dump([float(mean), float(std)], f)