import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class IXIT2Dataset2_5D(Dataset):
    def __init__(self, base_dir, mode="train", split_ratio=0.9, max_samples=None):
        self.mode = mode
        self.hr_dir = os.path.join(base_dir, 'HR')
        self.lr_dir = os.path.join(base_dir, 'LR')
        
        all_files = sorted([f for f in os.listdir(self.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if max_samples is not None:
            all_files = all_files[:max_samples]
            
        if mode in ["train", "val"]:
            split_idx = int(split_ratio * len(all_files))
            self.image_files = all_files[:split_idx] if mode == "train" else all_files[split_idx:]
        else:
            self.image_files = all_files
            
        print(f"[{mode.upper()}] distinct samples loaded: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def _get_tensor(self, img_name, is_hr=False):
        dir_path = self.hr_dir if is_hr else self.lr_dir
        img = Image.open(os.path.join(dir_path, img_name)).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        idx_prev = max(0, idx - 1)
        idx_next = min(len(self.image_files) - 1, idx + 1)
        
        t_prev = self._get_tensor(self.image_files[idx_prev], is_hr=False)
        t_mid = self._get_tensor(self.image_files[idx], is_hr=False)
        t_next = self._get_tensor(self.image_files[idx_next], is_hr=False)
        hr_tensor = self._get_tensor(self.image_files[idx], is_hr=True)
        
        lr_stacked = torch.cat([t_prev, t_mid, t_next], dim=0)
        return lr_stacked, t_mid, hr_tensor
