import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SuperResDataset(Dataset):
    def __init__(self, root_dir, split="train", buffer=50):
        self.image_dir = os.path.join(root_dir, 'Original')
        
        all_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        total_files = len(all_files)
        
        split_idx = int(0.9 * total_files)
        
        if split == "train":
            self.image_files = all_files[:split_idx]
        elif split == "test":
            start_idx = min(split_idx + buffer, total_files - 1)
            self.image_files = all_files[start_idx:]
            print(f"[{split.upper()}] distinct samples: {len(self.image_files)} (Buffer skipped {buffer} frames)")
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image_hr = Image.open(img_path).convert("RGB")
        image_hr = image_hr.resize((224, 224), resample=Image.BICUBIC)
        
        image_lr = image_hr.resize((56, 56), resample=Image.BICUBIC)
        
        hr_tensor = torch.from_numpy(np.array(image_hr)).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(np.array(image_lr)).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor
