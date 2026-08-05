import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Import your model from the existing files
from models import ResidualSR

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 10 * np.log10(1.0 / mse)

def preprocess_single_image(img_path):
    """Loads and prepares a single image for the SR model."""
    image_hr = Image.open(img_path).convert("RGB")
    image_hr = image_hr.resize((224, 224), resample=Image.BICUBIC)
    image_lr = image_hr.resize((56, 56), resample=Image.BICUBIC)
    
    hr_tensor = torch.from_numpy(np.array(image_hr)).permute(2, 0, 1).float() / 255.0
    lr_tensor = torch.from_numpy(np.array(image_lr)).permute(2, 0, 1).float() / 255.0
    
    return lr_tensor, hr_tensor

def run_multi_image_demo():
    # 1. Open File Dialog (Enabled for multiple files)
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    print("Opening file browser...")
    image_paths = filedialog.askopenfilenames(
        title="Select Endoscopy Images for SR Demo",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
    )
    
    if not image_paths:
        print("No images selected. Exiting demo.")
        return

    num_images = len(image_paths)
    print(f"Selected {num_images} image(s).")

    # 2. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "siglip_residual_super_resolution_final.pth"
    
    generator = ResidualSR(n_residual_blocks=8).to(device)
    if os.path.exists(weights_path):
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print(f"ERROR: {weights_path} not found. Please ensure the weights are in this directory.")
        return
    generator.eval()

    # 3. Setup Visualization Grid
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    col_titles = ["Ground Truth (HR)", "SigLIP-SR Output", "SigLIP-SR Error Map", "Bicubic Error Map"]
    ERROR_VMAX = 0.20 

    # Ensure axes is always a 2D array even if only 1 image is selected
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    def format_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0,0)
        ax.axis("off")

    # 4. Processing Loop
    for row, img_path in enumerate(image_paths):
        print(f"Processing ({row+1}/{num_images}): {os.path.basename(img_path)}")
        
        lr_tensor, hr_tensor = preprocess_single_image(img_path)
        pt_lr = lr_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            gen_hr = generator(pt_lr)
            bicubic = F.interpolate(pt_lr, scale_factor=4, mode='bicubic', align_corners=False)
                
        # Convert back to numpy for visualization
        ai_np = gen_hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bicubic_np = bicubic.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hr_np = hr_tensor.permute(1, 2, 0).cpu().numpy()
        
        psnr_ai = calculate_psnr(ai_np, hr_np)
        psnr_bi = calculate_psnr(bicubic_np, hr_np)
        
        # Calculate Error Maps
        err_ai_raw = np.mean(np.abs(ai_np - hr_np), axis=2)
        err_bi_raw = np.mean(np.abs(bicubic_np - hr_np), axis=2)
        
        # Endoscopy Circular Mask (Filters out the black camera borders)
        hr_gray = np.mean(hr_np, axis=2)
        mask = (hr_gray > 0.05).astype(np.float32)
        
        err_ai_masked = err_ai_raw * mask
        err_bi_masked = err_bi_raw * mask
        
        # Plot images into the grid
        axes[row, 0].imshow(np.clip(hr_np, 0, 1))
        format_ax(axes[row, 0])
        
        axes[row, 1].imshow(np.clip(ai_np, 0, 1))
        format_ax(axes[row, 1])
        
        im_ai_err = axes[row, 2].imshow(err_ai_masked, cmap='jet', vmin=0, vmax=ERROR_VMAX, interpolation='bilinear')
        format_ax(axes[row, 2])
        axes[row, 2].text(0.95, 0.05, f"{psnr_ai:.2f} dB", color='white', fontsize=14, ha='right', va='bottom', transform=axes[row, 2].transAxes)
        
        im_bi_err = axes[row, 3].imshow(err_bi_masked, cmap='jet', vmin=0, vmax=ERROR_VMAX, interpolation='bilinear')
        format_ax(axes[row, 3])
        axes[row, 3].text(0.95, 0.05, f"{psnr_bi:.2f} dB", color='white', fontsize=14, ha='right', va='bottom', transform=axes[row, 3].transAxes)
        
        # Add titles only to the top row
        if row == 0:
            for col in range(4):
                axes[row, col].set_title(col_titles[col], fontsize=14, pad=10)
                
    # Add a global Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_bi_err, cax=cbar_ax, label="Absolute Pixel Error")
        
    output_filename = "demo_results_grid.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Grid saved to '{output_filename}'")
    
    # Open the interactive plot window
    plt.show()

if __name__ == "__main__":
    run_multi_image_demo()
