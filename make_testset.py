# Use images selected by make_testdir.py to generate pickle file identical to challenge testset for testing before evaluating on server

from pathlib import Path
from PIL import Image
import dill as pickle
import torchvision.transforms as transforms
import numpy as np
from src.dataset import get_borders


testset_dir = "data/data_test"
outfile_dict = "files/my_testset_dict.pkl" # dictionary contains all relevant information about testset
outfile = "files/my_testset.pkl" # file with identical structure to challenge testset


testset_paths = sorted([str(p) for p in Path(testset_dir).rglob("*.jpg")])

borders_x = get_borders(len(testset_paths), seed=0)
borders_y = get_borders(len(testset_paths), seed=1)

rz = transforms.Resize((90,90))


known_arrays = []
input_arrays = []
target_arrays = []
sample_ids = []
for i, img_path in enumerate(testset_paths):
    img = Image.open(img_path)
    ar = np.array(rz(img))
    
    known_array = np.zeros_like(ar, dtype="uint8")
    border_x, border_y = borders_x[:,i], borders_y[:,i]
    
    known_array[border_x[0]:-border_x[1],border_y[0]:-border_y[1]] = 1
    target_array = ar[known_array==0]
    ar[:border_x[0]] = 0
    ar[-border_x[1]:] = 0
    ar[:,:border_y[0]] = 0
    ar[:,-border_y[1]:] = 0
    
    known_arrays.append(known_array)
    input_arrays.append(ar)
    target_arrays.append(target_array)
    sample_ids.append(i)
    
    
testset = {
    "input_arrays": tuple(input_arrays),
    "known_arrays": tuple(known_arrays),
    "target_arrays": tuple(target_arrays),
    "borders_x": borders_x,
    "borders_y": borders_y,
    "sample_ids": tuple(sample_ids)
}


with open(outfile_dict, "wb") as f:
    pickle.dump(testset, f)
with open(outfile, "wb") as f:
    pickle.dump(target_arrays, f)