import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from config import CONFIG
from dataset import IXIT2Dataset2_5D
from models import RRDBNet_25D
from metrics import calculate_ssim_np

torch.backends.cudnn.benchmark = True 

CONFIG['MODEL_WEIGHTS'] = "Ultimate_RRDB_25D_ixi_best.pth"
CONFIG['BATCH_SIZE'] = 1 

def run_benchmark_only():
    print("\n" + "="*70)
    print("RUNNING FINAL IXI BENCHMARK (WITH STANDARD DEVIATION)")
    print("="*70)
    
    device = torch.device(CONFIG['DEVICE'])
    
    generator = RRDBNet_25D().to(device)
    if not os.path.exists(CONFIG['MODEL_WEIGHTS']):
        print(f"ERROR: Could not find weights at {CONFIG['MODEL_WEIGHTS']}")
        return
        
    generator.load_state_dict(torch.load(CONFIG['MODEL_WEIGHTS'], map_location=device, weights_only=True))
    generator.eval()
    print(f"Successfully loaded weights: {CONFIG['MODEL_WEIGHTS']}")
    
    print("Loading LPIPS VGG Model...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    test_ds = IXIT2Dataset2_5D(CONFIG['TEST_BASE_DIR'], mode="test", max_samples=None)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
    
    psnr_ai_list, psnr_bi_list = [], []
    ssim_ai_list, ssim_bi_list = [], []
    lpips_ai_list, lpips_bi_list = [], []
    
    with torch.no_grad():
        for lr_stacked, lr_mid, hr_imgs in tqdm(test_loader, desc="Evaluating", leave=True):
            lr_stacked = lr_stacked.to(device, non_blocking=True)
            lr_mid = lr_mid.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda' if 'cuda' in CONFIG['DEVICE'] else 'cpu'):
                gen_hr = generator(lr_stacked, lr_mid)
                
            bicubic = F.interpolate(lr_mid, scale_factor=4, mode='bicubic', align_corners=False)
            
            mse_ai = torch.clamp(F.mse_loss(gen_hr, hr_imgs), min=1e-10) 
            mse_bi = torch.clamp(F.mse_loss(bicubic, hr_imgs), min=1e-10)
            psnr_ai_list.append((10 * torch.log10(1 / mse_ai)).item())
            psnr_bi_list.append((10 * torch.log10(1 / mse_bi)).item())
            
            ssim_ai_list.append(calculate_ssim_np(gen_hr, hr_imgs))
            ssim_bi_list.append(calculate_ssim_np(bicubic, hr_imgs))
            
            lpips_ai_list.append(lpips_fn((gen_hr * 2 - 1), (hr_imgs * 2 - 1)).item())
            lpips_bi_list.append(lpips_fn((bicubic * 2 - 1), (hr_imgs * 2 - 1)).item())

    metrics = {
        "PSNR_BI": (np.mean(psnr_bi_list), np.std(psnr_bi_list)),
        "PSNR_AI": (np.mean(psnr_ai_list), np.std(psnr_ai_list)),
        "SSIM_BI": (np.mean(ssim_bi_list), np.std(ssim_bi_list)),
        "SSIM_AI": (np.mean(ssim_ai_list), np.std(ssim_ai_list)),
        "LPIPS_BI": (np.mean(lpips_bi_list), np.std(lpips_bi_list)),
        "LPIPS_AI": (np.mean(lpips_ai_list), np.std(lpips_ai_list)),
    }
    
    print("\n" + "="*70)
    print(f"{'Metric':<10} | {'Bicubic Baseline (Mean ± Std)':<30} | {'Ultimate RRDB-SR (Mean ± Std)'}")
    print("-" * 70)
    print(f"{'PSNR':<10} | {metrics['PSNR_BI'][0]:.2f} ± {metrics['PSNR_BI'][1]:.2f} dB{'':<10} | {metrics['PSNR_AI'][0]:.2f} ± {metrics['PSNR_AI'][1]:.2f} dB")
    print(f"{'SSIM':<10} | {metrics['SSIM_BI'][0]:.4f} ± {metrics['SSIM_BI'][1]:.4f}{'':<13} | {metrics['SSIM_AI'][0]:.4f} ± {metrics['SSIM_AI'][1]:.4f}")
    print(f"{'LPIPS':<10} | {metrics['LPIPS_BI'][0]:.4f} ± {metrics['LPIPS_BI'][1]:.4f}{'':<13} | {metrics['LPIPS_AI'][0]:.4f} ± {metrics['LPIPS_AI'][1]:.4f}")
    print("-" * 70)
    
    gain_psnr = metrics['PSNR_AI'][0] - metrics['PSNR_BI'][0]
    gain_ssim = metrics['SSIM_AI'][0] - metrics['SSIM_BI'][0]
    gain_lpips = metrics['LPIPS_BI'][0] - metrics['LPIPS_AI'][0] 
    
    print(f"NET GAIN   | PSNR: +{gain_psnr:.2f} dB  | SSIM: +{gain_ssim:.4f}  | LPIPS: +{gain_lpips:.4f} improvement")
    print("="*70)

if __name__ == "__main__":
    if not os.path.exists(CONFIG['TEST_BASE_DIR']):
        print(f"ERROR: Test dataset directory '{CONFIG['TEST_BASE_DIR']}' not found.")
    else:
        run_benchmark_only()
