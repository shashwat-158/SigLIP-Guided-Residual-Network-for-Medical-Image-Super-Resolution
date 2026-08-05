import torch

CONFIG = {
    'DATA_PATH': './CVC-ClinicDB/PNG', 
    'EPOCHS': 20,                      
    'BATCH_SIZE': 8,
    'LR': 5e-4,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'BUFFER_SIZE': 50,                 
    'MODEL_NAME': "google/siglip2-so400m-patch14-224" 
}
