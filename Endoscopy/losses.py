import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

class SigLIPPerceptualLoss(nn.Module): 
    def __init__(self, model_name):
        super().__init__()
        print(f" > Loading Frozen SigLIP: {model_name}...")
        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.target_layer = 12 

    def forward(self, generated_img, target_img):
        out_gen = self.backbone(pixel_values=generated_img, output_hidden_states=True)
        feat_gen = out_gen.hidden_states[self.target_layer]
        
        with torch.no_grad():
            out_target = self.backbone(pixel_values=target_img, output_hidden_states=True)
            feat_target = out_target.hidden_states[self.target_layer]
        
        return F.mse_loss(feat_gen, feat_target)
