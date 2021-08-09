from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def get_mask_vals(avg_border_size = 12):
    """get normalized values for 0/1 mask"""
    mask_mean = (90 - avg_border_size)**2 / 90**2
    mask_std = (mask_mean - mask_mean**2)**(.5)
    mask_val_zero = (0 - mask_mean) / mask_std
    mask_val_one = (1 - mask_mean) / mask_std
    return mask_val_zero, mask_val_one


def get_borders(n, seed=None):
    rng = np.random.default_rng(seed=seed)
    borders_a = rng.integers(5, 9, (n,))
    borders_b = rng.integers(5, (15 - borders_a), (n,))
    edge = np.zeros((2,n), dtype=int)
    edge[rng.choice([0, 1], size=(n,)), np.arange(n)] = 1
    return np.stack([borders_a, borders_b])[edge, np.arange(n)]


def denorm(img_ar, mean, std):
    tf = transforms.Compose([
        transforms.Normalize((-1 * mean / std), (1.0 / std)),
        transforms.Lambda(lambda x_: torch.clamp(x_ * 255, 0, 255))
    ])
    return tf(img_ar)
    


class ImageDS(Dataset):
    
    def __init__(self, root_dir: str, debug=False):
        
        self.root_dir = Path(root_dir)
        self.file_paths = sorted([str(p) for p in self.root_dir.rglob("*.jpg")])
        
        if debug:
            self.file_paths = self.file_paths[:100]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        image_data = self.file_paths[idx]
        
        return image_data, idx


class BorderPredictionDS(Dataset):
        
    def __init__(self, dataset: Dataset, ds_stats: tuple = (0, 1), transform_chain: transforms.Compose = None, border_mode: str = "max"):
        
        self.dataset = dataset
        self.border_mode = border_mode
        if self.border_mode == "max":
            self.border_dummy = tuple([9,9])
        elif self.border_mode == "fix":
            self.borders_x = self.get_borders_(len(dataset), seed=0)
            self.borders_y = self.get_borders_(len(dataset), seed=1)
        elif self.border_mode == "rand":
            self.n_updates = 0
            self.reset_at = 10000
            self.borders_x, self.borders_y = self.reset_random_borders(self.reset_at)
        else:
            raise Exception(f"invalid border mode {self.border_mode}")
            
        self.get_mask_vals_(avg_border_size = 12)

        self.ds_mean, self.ds_std = ds_stats
        # self.ds_val_zero = - self.ds_mean / self.ds_std

        norm = transforms.Normalize((self.ds_mean,), (self.ds_std,))
        transform_chain.transforms.insert(len(transform_chain.transforms), norm)
        self.transform_chain = transform_chain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        img_data, idx_ = self.dataset.__getitem__(idx)
            
        img = Image.open(img_data)
        
        ar = self.transform_chain(img)
        
        if self.border_mode == "max":
            border_x, border_y = self.border_dummy, self.border_dummy
        elif self.border_mode == "fix":
            border_x, border_y = self.borders_x[idx], self.borders_y[idx]
        elif self.border_mode == "rand":
            border_x, border_y = self.borders_x[self.n_updates], self.borders_y[self.n_updates]
            if self.n_updates >= (self.reset_at - 1):
                self.borders_x, self.borders_y = self.reset_random_borders(self.reset_at)
                self.n_updates = 0
            else:
                self.n_updates += 1
        else:
            raise Exception(f"invalid border mode {self.border_mode}")
        
        x, y, mask = self.prepare_arrays(ar, border_x, border_y)
        
        return x, y, mask, idx_
    
    @staticmethod
    def get_borders_(n, seed=None):
        borders = get_borders(n, seed)
        return list(map(tuple, borders.T))
    
    def prepare_arrays(self, image_array: torch.Tensor, border_x: tuple, border_y: tuple):

        mask = torch.zeros_like(image_array, dtype=torch.bool)
        mask[:,border_x[0]:-border_x[1],border_y[0]:-border_y[1]] = True        
        known_array = torch.full_like(image_array, self.mask_val_zero)
        known_array[mask] = self.mask_val_one
        
        target_array = image_array[~mask]

        image_array[~mask] = self.ds_mean
        
        return torch.cat([image_array, known_array]), target_array, mask
    
    def reset_random_borders(self, n):
        return self.get_borders_(self.reset_at), self.get_borders_(self.reset_at)
    
    def get_mask_vals_(self, avg_border_size = 12):
        """get normalized values for 0/1 mask"""
        self.mask_val_zero, self.mask_val_one = get_mask_vals(avg_border_size = 12)
        return self.mask_val_zero, self.mask_val_one
    
    
class Testset(Dataset):    
    
    def __init__(self, filename: str, mean: float, std: float):
        
        self.filename = Path(filename)
        self.ds_mean = mean
        self.ds_std = std
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        with open(self.filename, "rb") as f:
            self.data = pickle.load(f)
            
        self.get_mask_vals_(avg_border_size = 12)
        
        # self.ds_val_zero = - self.ds_mean / self.ds_std  
        
    def __len__(self):
        return len(self.data["input_arrays"])

    def __getitem__(self, idx):
        
        input_array = self.data["input_arrays"][idx]
        mask = self.data["known_arrays"][idx]
               
        input_array = self.transform(input_array).squeeze(0)
        input_array[~mask] = self.ds_mean
        known_array = torch.full_like(input_array, self.mask_val_zero)
        known_array[mask] = self.mask_val_one
        
        ar = torch.stack([input_array, known_array], dim=0).float()
        
        return ar, mask
    
    def denormalize(self, x: torch.tensor):
        return denorm(x, self.ds_mean, self.ds_std)
    
    def get_mask_vals_(self, avg_border_size = 12):
        """get normalized values for 0/1 mask"""
        self.mask_val_zero, self.mask_val_one = get_mask_vals(avg_border_size = 12)
        return self.mask_val_zero, self.mask_val_one