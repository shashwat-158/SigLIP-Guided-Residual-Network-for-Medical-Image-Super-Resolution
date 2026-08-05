import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import SuperResDataset
from models import ResidualSR
from losses import SigLIPPerceptualLoss

def run_full_pipeline():
    print(f"=== STARTING PIPELINE ON {CONFIG['DEVICE']} ===")
    
    generator = ResidualSR().to(CONFIG['DEVICE'])
    perceptual_loss = SigLIPPerceptualLoss(CONFIG['MODEL_NAME']).to(CONFIG['DEVICE'])
    optimizer = optim.AdamW(generator.parameters(), lr=CONFIG['LR'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_ds = SuperResDataset(CONFIG['DATA_PATH'], split="train", buffer=CONFIG['BUFFER_SIZE'])
    test_ds = SuperResDataset(CONFIG['DATA_PATH'], split="test", buffer=CONFIG['BUFFER_SIZE'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"\nTraining for {CONFIG['EPOCHS']} epochs...")
    for epoch in range(CONFIG['EPOCHS']):
        generator.train()
        total_loss = 0
        
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(CONFIG['DEVICE']), hr_imgs.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            gen_hr = generator(lr_imgs)
            
            l_pixel = F.l1_loss(gen_hr, hr_imgs)
            l_feat = perceptual_loss(gen_hr, hr_imgs)
            loss = l_pixel + (0.05 * l_feat)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | Loss: {total_loss/len(train_loader):.5f}")

    print("\n" + "="*40)
    print("RUNNING FINAL NON-INFLATED BENCHMARK")
    print("="*40)
    
    generator.eval()
    psnr_ai_accum = 0.0
    psnr_bi_accum = 0.0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs, hr_imgs = lr_imgs.to(CONFIG['DEVICE']), hr_imgs.to(CONFIG['DEVICE'])
            
            gen_hr = generator(lr_imgs)
            mse_ai = F.mse_loss(gen_hr, hr_imgs)
            psnr_ai = 10 * torch.log10(1 / mse_ai)
            psnr_ai_accum += psnr_ai.item()
            
            bicubic = F.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False)
            mse_bi = F.mse_loss(bicubic, hr_imgs)
            psnr_bi = 10 * torch.log10(1 / mse_bi)
            psnr_bi_accum += psnr_bi.item()

    avg_ai = psnr_ai_accum / len(test_loader)
    avg_bi = psnr_bi_accum / len(test_loader)
    
    print(f"Baseline Bicubic PSNR: {avg_bi:.2f} dB")
    print(f"Your SigLIP-SR PSNR:   {avg_ai:.2f} dB")
    print(f"Real Gain:             {avg_ai - avg_bi:+.2f} dB")
    print("="*40)
    
    if avg_ai > avg_bi + 0.5:
        print("VERDICT: SUCCESS. Model has learned valid Super-Resolution.")
    else:
        print("VERDICT: CONVERGENCE ISSUE. Try more epochs or check data.")

    save_path = "siglip_residual_super_resolution_final.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    if not os.path.exists(os.path.join(CONFIG['DATA_PATH'], 'Original')):
        print(f"ERROR: Dataset not found at {os.path.join(CONFIG['DATA_PATH'], 'Original')}")
        print("Please create the folder and add CVC-ClinicDB images.")
    else:
        run_full_pipeline()
