import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import lpips

# Import your model from the existing files
from models import RRDBNet_25D

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(1.0 / mse)

def load_existing_pair(hr_path):
    """Loads pre-existing HR and LR pairs based on folder structure, starting from HR."""
    
    # Automatically map the HR path back to the LR path
    # Accounts for both Windows (\) and Unix (/) path separators
    lr_path = hr_path.replace('/HR/', '/LR/').replace('\\HR\\', '\\LR\\')
    
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"\n[!] ERROR: Could not find matching LR image.\nExpected it at: {lr_path}")

    # Load images exactly as they are in the folders
    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")
    
    # Convert directly to tensors [0, 1] without resizing
    lr_tensor = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0
    hr_tensor = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
    
    return lr_tensor, hr_tensor

def run_publication_demo():
    # 1. Open File Dialog
    root = tk.Tk()
    root.withdraw() 
    print("Opening file browser...")
    
    # Instruct the user to select from the HR folder
    image_paths = filedialog.askopenfilenames(
        title="Select HR MRI Slices (Select consecutive slices from the HR folder)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
    )
    
    if not image_paths:
        print("No images selected. Exiting demo.")
        return

    # Sort paths alphabetically to ensure proper spatial/temporal slice ordering
    image_paths = sorted(list(image_paths))
    num_images = len(image_paths)
    print(f"Selected {num_images} HR image(s). Corresponding LR images will be loaded automatically.")

    # 2. Setup Device and Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "Ultimate_RRDB_25D_ixi_best.pth"
    
    generator = RRDBNet_25D(in_channels=9, out_channels=3, num_blocks=4).to(device)
    if os.path.exists(weights_path):
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("Successfully loaded Ultimate RRDB-SR weights.")
    else:
        print(f"ERROR: '{weights_path}' not found. Ensure it is in the root directory.")
        return
        
    generator.eval()

    print("Loading LPIPS VGG Model for perceptual evaluation...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # 3. Load all selected HR/LR pairs
    lr_tensors, hr_tensors = [], []
    for path in image_paths:
        try:
            lr_t, hr_t = load_existing_pair(path)
            lr_tensors.append(lr_t)
            hr_tensors.append(hr_t)
        except FileNotFoundError as e:
            print(e)
            return # Abort if the folder structure isn't correct

    # 4. Setup Visualization Grid
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images), gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
    col_titles = ["LR Output", "HR Output", "SigLIP-SR Output", "SigLIP-SR Error Map"]
    ERROR_VMAX = 0.25 

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    def format_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0,0)
        ax.axis("off")

    # 5. Processing Loop
    for row in range(num_images):
        print(f"Processing ({row+1}/{num_images}): {os.path.basename(image_paths[row])}")
        
        # Determine 2.5D adjacent slices (fallback to current slice if at boundaries)
        idx_prev = max(0, row - 1)
        idx_next = min(num_images - 1, row + 1)
        
        t_prev = lr_tensors[idx_prev]
        t_mid = lr_tensors[row]
        t_next = lr_tensors[idx_next]
        hr_tensor = hr_tensors[row]
        
        # Stack 3 slices (9 channels) for the RRDBNet_25D input
        lr_stacked = torch.cat([t_prev, t_mid, t_next], dim=0).unsqueeze(0).to(device)
        lr_mid_pt = t_mid.unsqueeze(0).to(device)
        hr_pt = hr_tensor.unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            gen_hr = generator(lr_stacked, lr_mid_pt)
            val_lpips_ai = lpips_fn((gen_hr * 2.0 - 1.0), (hr_pt * 2.0 - 1.0)).item()
                
        # Convert back to numpy
        lr_np = t_mid.permute(1, 2, 0).cpu().numpy()
        ai_np = gen_hr.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        hr_np = hr_tensor.permute(1, 2, 0).cpu().numpy()
        
        psnr_ai = calculate_psnr(ai_np, hr_np)
        err_ai = np.mean(np.abs(ai_np - hr_np), axis=2)
        
        # Plotting
        axes[row, 0].imshow(np.clip(lr_np, 0, 1))
        format_ax(axes[row, 0])
        
        axes[row, 1].imshow(np.clip(hr_np, 0, 1))
        format_ax(axes[row, 1])
        
        axes[row, 2].imshow(np.clip(ai_np, 0, 1))
        format_ax(axes[row, 2])
        
        im_ai_err = axes[row, 3].imshow(err_ai, cmap='jet', vmin=0, vmax=ERROR_VMAX, interpolation='bilinear')
        format_ax(axes[row, 3])
        
        # Display Metrics in the style of your original code
        axes[row, 3].text(0.95, 0.05, f"PSNR: {psnr_ai:.2f}\nLPIPS: {val_lpips_ai:.4f}", 
                          color='white', fontsize=14, 
                          ha='right', va='bottom', transform=axes[row, 3].transAxes)
        
        # Title only top row
        if row == 0:
            for col in range(4):
                axes[row, col].set_title(col_titles[col], fontsize=14, pad=10)
                
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_ai_err, cax=cbar_ax, label="Absolute Pixel Error")
        
    output_filename = "top_psnr_error_maps.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Success! Saved top-tier publication error maps to '{output_filename}'")
    
    # Open interactive window
    plt.show()

if __name__ == "__main__":
    run_publication_demo()
