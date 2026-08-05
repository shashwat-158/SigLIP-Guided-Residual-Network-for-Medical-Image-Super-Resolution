from skimage.metrics import structural_similarity as ssim

def calculate_ssim_np(img_tensor, ref_tensor):
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ref_np = ref_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ssim(img_np, ref_np, data_range=1.0, channel_axis=2)
