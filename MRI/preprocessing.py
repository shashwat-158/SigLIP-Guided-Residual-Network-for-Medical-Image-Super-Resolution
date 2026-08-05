import os
import glob
import numpy as np
import nibabel as nib
import cv2
import random

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DIR = "./IXI-T2"       # Folder containing .nii or .nii.gz files
BASE_SAVE_DIR = "./dataset"

# The paper explicitly uses 300 subjects for training and 176 for testing
TRAIN_SPLIT_COUNT = 300 
TEST_SPLIT_COUNT = 176
# The paper retains 80 standardized slices per subject
SLICES_PER_SUBJECT = 80 
SCALE_FACTOR = 4

# Create directories
for split in ['train', 'test']:
    os.makedirs(os.path.join(BASE_SAVE_DIR, split, 'HR'), exist_ok=True)
    os.makedirs(os.path.join(BASE_SAVE_DIR, split, 'LR'), exist_ok=True)

# ==========================================
# DEGRADATION MODEL (As per MICCAI Paper)
# ==========================================
def degrade_image(hr_img, scale=4):
    """
    Applies Gaussian blur, additive white Gaussian noise, 
    and bicubic downsampling to simulate real-world MRI degradation.
    """
    # 1. General Gaussian blur kernel
    blurred = cv2.GaussianBlur(hr_img, (5, 5), sigmaX=1.2, sigmaY=1.2)
    
    # 2. Additive White Gaussian Noise (AWGN)
    noise = np.random.normal(0, 15, blurred.shape) # mean=0, std=15 (tune as needed)
    noisy_img = np.clip(blurred + noise, 0, 255).astype(np.float32)
    
    # 3. Bicubic downsampling
    h, w = noisy_img.shape[:2]
    lr_img = cv2.resize(noisy_img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    
    return np.clip(lr_img, 0, 255).astype(np.uint8)

# ==========================================
# PROCESSING LOOP
# ==========================================
nii_files = glob.glob(os.path.join(SOURCE_DIR, "*.nii*"))
print(f"Found {len(nii_files)} volumes in {SOURCE_DIR}")

# 1. SHUFFLE AND SPLIT BY SUBJECT (Fixing Data Leakage)
random.seed(42) # For reproducibility
random.shuffle(nii_files)

train_files = nii_files[:TRAIN_SPLIT_COUNT]
test_files = nii_files[TRAIN_SPLIT_COUNT : TRAIN_SPLIT_COUNT + TEST_SPLIT_COUNT]

datasets = [('train', train_files), ('test', test_files)]
total_processed = 0

for split_name, files in datasets:
    print(f"\nProcessing {split_name.upper()} set ({len(files)} subjects)...")
    save_dir_hr = os.path.join(BASE_SAVE_DIR, split_name, 'HR')
    save_dir_lr = os.path.join(BASE_SAVE_DIR, split_name, 'LR')
    
    for filepath in files:
        try:
            img_obj = nib.load(filepath)
            img_data = img_obj.get_fdata()
            
            # Robust Normalization to [0, 255] for PNG saving
            lower = np.percentile(img_data, 0.5)
            upper = np.percentile(img_data, 99.5)
            img_data = np.clip(img_data, lower, upper)
            
            if upper - lower != 0:
                img_data = (img_data - lower) / (upper - lower) * 255
            else:
                img_data[:] = 0
                
            img_data = img_data.astype(np.uint8)

            # Extract middle 80 slices (Fixing Slice Range)
            num_slices = img_data.shape[2]
            start_idx = max(0, (num_slices - SLICES_PER_SUBJECT) // 2)
            end_idx = min(num_slices, start_idx + SLICES_PER_SUBJECT)

            for i in range(start_idx, end_idx):
                slice_img = img_data[:, :, i]
                
                # Save High-Res (HR) at 256x256
                hr_img = cv2.resize(slice_img, (256, 256), interpolation=cv2.INTER_AREA)
                
                # Apply Paper's specific Degradation Pipeline for LR
                lr_img = degrade_image(hr_img, scale=SCALE_FACTOR)
                
                # Save Files
                file_id = os.path.basename(filepath).split('.')[0]
                filename = f"{file_id}_slice{i}.png"
                
                cv2.imwrite(os.path.join(save_dir_hr, filename), hr_img)
                cv2.imwrite(os.path.join(save_dir_lr, filename), lr_img)
                
                total_processed += 1
                if total_processed % 500 == 0: 
                    print(f"Processed {total_processed} slices...", end='\r')

        except Exception as e:
            print(f"\nError processing {filepath}: {e}")

print(f"\n\nDone! Successfully extracted {total_processed} paired slices.")
print("Subject-level split completed successfully to prevent data leakage.")
