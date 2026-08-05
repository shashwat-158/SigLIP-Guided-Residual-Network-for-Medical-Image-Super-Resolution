import os
import gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from config import CONFIG
from dataset import IXIT2Dataset2_5D
from models import RRDBNet_25D
from losses import SigLIPPerceptualLoss
from metrics import calculate_ssim_np

torch.backends.cudnn.benchmark = True 

def run_ixi_pipeline():
    print(f"=== STARTING 12-HOUR 24K RRDB+2.5D PIPELINE ON {CONFIG['DEVICE']} ===")
    
    generator = RRDBNet_25D().to(CONFIG['DEVICE'])
    perceptual_loss = SigLIPPerceptualLoss(CONFIG['MODEL_NAME']).to(CONFIG['DEVICE'])
    
    print(" > Loading LPIPS VGG Model...")
    lpips_fn = lpips.LPIPS(net='vgg').to(CONFIG['DEVICE'])
    
    optimizer = optim.AdamW(generator.parameters(), lr=CONFIG['LR'], fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['TOTAL_EPOCHS'], eta_min=1e-6)
    
    train_ds = IXIT2Dataset2_5D(CONFIG['TRAIN_BASE_DIR'], mode="train", max_samples=CONFIG['MAX_SAMPLES']) 
    val_ds = IXIT2Dataset2_5D(CONFIG['TRAIN_BASE_DIR'], mode="val", max_samples=CONFIG['MAX_SAMPLES']) 
    test_ds = IXIT2Dataset2_5D(CONFIG['TEST_BASE_DIR'], mode="test", max_samples=None)
    
    cpu_workers = min(os.cpu_count(), 8) 
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=cpu_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    best_val_lpips = float('inf') 
    best_model_path = "Ultimate_RRDB_25D_ixi_best.pth"
    
    scaler = torch.amp.GradScaler('cuda')

    print(f"\nTraining for {CONFIG['TOTAL_EPOCHS']} total epochs...")
    for epoch in range(CONFIG['TOTAL_EPOCHS']):
        
        if epoch == CONFIG['STAGE1_EPOCHS']:
            print("\n" + "*"*60)
            print(">>> STAGE 2 INITIATED: FULL PERCEPTUAL OVERDRIVE <<<")
            print("*"*60 + "\n")
            
        is_stage_2 = epoch >= CONFIG['STAGE1_EPOCHS']
        
        generator.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['TOTAL_EPOCHS']} [TRAIN]", leave=False)
        
        for lr_stacked, lr_mid, hr_imgs in train_pbar:
            lr_stacked = lr_stacked.to(CONFIG['DEVICE'], non_blocking=True)
            lr_mid = lr_mid.to(CONFIG['DEVICE'], non_blocking=True)
            hr_imgs = hr_imgs.to(CONFIG['DEVICE'], non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                gen_hr = generator(lr_stacked, lr_mid)
                l_pixel = F.l1_loss(gen_hr, hr_imgs)
                
                if not is_stage_2:
                    loss = l_pixel
                else:
                    gen_hr_critic = F.interpolate(gen_hr, size=(384, 384), mode='bicubic', align_corners=False)
                    hr_critic = F.interpolate(hr_imgs, size=(384, 384), mode='bicubic', align_corners=False)
                    l_feat = perceptual_loss(gen_hr_critic, hr_critic)
                    loss = (0.01 * l_pixel) + l_feat
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            del gen_hr, l_pixel, loss
            if is_stage_2:
                del gen_hr_critic, hr_critic, l_feat
                
        torch.cuda.empty_cache()
        gc.collect()
            
        scheduler.step()
        
        generator.eval()
        val_lpips_accum = 0.0
        with torch.no_grad():
            for lr_stacked, lr_mid, hr_imgs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['TOTAL_EPOCHS']} [VAL]", leave=False):
                lr_stacked = lr_stacked.to(CONFIG['DEVICE'], non_blocking=True)
                lr_mid = lr_mid.to(CONFIG['DEVICE'], non_blocking=True)
                hr_imgs = hr_imgs.to(CONFIG['DEVICE'], non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    gen_hr = generator(lr_stacked, lr_mid)
                    val_lpips = lpips_fn((gen_hr * 2 - 1), (hr_imgs * 2 - 1)).mean()
                    
                val_lpips_accum += val_lpips.item() * hr_imgs.size(0) 
                
        avg_val_lpips = val_lpips_accum / len(val_ds)
        print(f"Epoch {epoch+1}/{CONFIG['TOTAL_EPOCHS']} | Train Loss: {total_loss/len(train_loader):.5f} | Val LPIPS: {avg_val_lpips:.4f}")

        if avg_val_lpips < best_val_lpips:
            best_val_lpips = avg_val_lpips
            torch.save(generator.state_dict(), best_model_path)
            print(f"  -> New best model saved! (LPIPS: {best_val_lpips:.4f})")

    print("\n" + "="*60)
    print("RUNNING FINAL IXI BENCHMARK")
    print("="*60)
    
    generator.load_state_dict(torch.load(best_model_path, weights_only=True))
    generator.eval()
    
    psnr_ai_accum, psnr_bi_accum = 0.0, 0.0
    ssim_ai_accum, ssim_bi_accum = 0.0, 0.0
    lpips_ai_accum, lpips_bi_accum = 0.0, 0.0
    
    with torch.no_grad():
        for lr_stacked, lr_mid, hr_imgs in tqdm(test_loader, desc="Evaluating", leave=True):
            lr_stacked = lr_stacked.to(CONFIG['DEVICE'], non_blocking=True)
            lr_mid = lr_mid.to(CONFIG['DEVICE'], non_blocking=True)
            hr_imgs = hr_imgs.to(CONFIG['DEVICE'], non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                gen_hr = generator(lr_stacked, lr_mid)
                
            bicubic = F.interpolate(lr_mid, scale_factor=4, mode='bicubic', align_corners=False)
            
            mse_ai = F.mse_loss(gen_hr, hr_imgs)
            mse_ai = torch.clamp(mse_ai, min=1e-10) 
            psnr_ai_accum += (10 * torch.log10(1 / mse_ai)).item()
            
            mse_bi = F.mse_loss(bicubic, hr_imgs)
            mse_bi = torch.clamp(mse_bi, min=1e-10)
            psnr_bi_accum += (10 * torch.log10(1 / mse_bi)).item()
            
            ssim_ai_accum += calculate_ssim_np(gen_hr, hr_imgs)
            ssim_bi_accum += calculate_ssim_np(bicubic, hr_imgs)
            
            lpips_ai_accum += lpips_fn((gen_hr * 2 - 1), (hr_imgs * 2 - 1)).item()
            lpips_bi_accum += lpips_fn((bicubic * 2 - 1), (hr_imgs * 2 - 1)).item()

    num_test = len(test_loader)
    avg_psnr_ai = psnr_ai_accum / num_test
    avg_psnr_bi = psnr_bi_accum / num_test
    avg_ssim_ai = ssim_ai_accum / num_test
    avg_ssim_bi = ssim_bi_accum / num_test
    avg_lpips_ai = lpips_ai_accum / num_test
    avg_lpips_bi = lpips_bi_accum / num_test
    
    print(f"\nBaseline Bicubic : PSNR: {avg_psnr_bi:.2f} dB | SSIM: {avg_ssim_bi:.4f} | LPIPS: {avg_lpips_bi:.4f}")
    print(f"Ultimate RRDB-SR : PSNR: {avg_psnr_ai:.2f} dB | SSIM: {avg_ssim_ai:.4f} | LPIPS: {avg_lpips_ai:.4f}")
    
    print(f"Real Gain        : PSNR: +{avg_psnr_ai - avg_psnr_bi:.2f} dB | SSIM: +{avg_ssim_ai - avg_ssim_bi:.4f} | LPIPS: +{avg_lpips_bi - avg_lpips_ai:.4f} improvement")
    print("="*60)

if __name__ == "__main__":
    if not os.path.exists(CONFIG['TRAIN_BASE_DIR']):
        print("ERROR: IXI Dataset directories not found.")
    else:
        run_ixi_pipeline()
