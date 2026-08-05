import torch

CONFIG = {
    'TRAIN_BASE_DIR': './dataset/train', 
    'TEST_BASE_DIR': './dataset/test',
    'STAGE1_EPOCHS': 15,  
    'STAGE2_EPOCHS': 35,  
    'BATCH_SIZE': 4,
    'LR': 2e-4, 
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'MODEL_NAME': "google/siglip2-so400m-patch14-384", 
    'MAX_SAMPLES': 24000  
}
CONFIG['TOTAL_EPOCHS'] = CONFIG['STAGE1_EPOCHS'] + CONFIG['STAGE2_EPOCHS']
