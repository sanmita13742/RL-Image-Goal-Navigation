import sys
from pathlib import Path

import torch
import timm

def main():
    device = torch.device("cpu")
    model_name = "vit_small_patch16_dinov3"
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    
    x = torch.randn(1, 3, 448, 784)
    with torch.no_grad():
        features = model.forward_features(x)
        print(f"forward_features(x) shape: {features.shape}")
        
        out = model(x)
        print(f"model(x) shape: {out.shape}")
        
        # Check if model has a head
        print(f"Head: {getattr(model, 'head', None)}")
        print(f"Global pool: {getattr(model, 'global_pool', None)}")
        
        # Let's see if features[:, 0] is the same as model.forward_head(features, pre_logits=True)
        if hasattr(model, 'forward_head'):
            head_out = model.forward_head(features, pre_logits=True)
            print(f"forward_head(features, pre_logits=True) shape: {head_out.shape}")
            diff = (head_out - features[:, 0]).abs().max()
            print(f"Diff between forward_head(pre_logits=True) and features[:, 0]: {diff}")
            
if __name__ == '__main__':
    main()
