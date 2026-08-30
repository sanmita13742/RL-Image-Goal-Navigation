"""
scripts/verify_dinov3.py
============================================================
Live verification of DINOv3 model on a real dataset image.

Performs:
1. Loads vit_small_patch16_dinov3
2. Checks frozen state
3. Loads a real image (320x240) and applies preprocessing
4. Runs forward pass
5. Prints exact tensor shapes, dtypes, tokens, and device
"""

import sys
import timm
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

def main():
    print("============================================================")
    print("DINOv3 LIVE VERIFICATION")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n1. Model Loading:")
    model_name = "vit_small_patch16_dinov3"
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    
    frozen = all(not p.requires_grad for p in model.parameters())
    print(f"Model is fully frozen (requires_grad=False): {frozen}")

    print("\n2. Data Preprocessing:")
    # Get a real image
    image_path = Path("dataset/20260811_234812/segment_000/rgb/000000.png")
    if not image_path.exists():
        print(f"ERROR: Could not find {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    print(f"Original image size: {img.size} (W x H)")

    transform = transforms.Compose([
        transforms.Resize((448, 784)),
        transforms.ToTensor()
    ])
    
    x = transform(img).unsqueeze(0).to(device)
    print(f"Preprocessed input tensor shape: {tuple(x.shape)}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input pixel range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print("NOTE: No ImageNet normalization (mean/std) is applied.")

    print("\n3. Model Forward Pass:")
    with torch.no_grad():
        head_out = model(x)
        feats = model.forward_features(x)

    print(f"model(x) [Classification Head] shape: {tuple(head_out.shape)}")
    print(f"model.forward_features(x) shape:      {tuple(feats.shape)}")
    print(f"Features dtype: {feats.dtype}")

    print("\n4. Token Analysis:")
    n_total_tokens = feats.shape[1]
    n_special = 5
    n_patch = n_total_tokens - n_special
    embed_dim = feats.shape[2]

    print(f"Total tokens:    {n_total_tokens}")
    print(f"Special tokens:  {n_special} (1 CLS + 4 registers typically for DINOv2/v3)")
    print(f"Patch tokens:    {n_patch}")
    print(f"Embedding dim:   {embed_dim}")

    expected_h = 448 // 16
    expected_w = 784 // 16
    print(f"\nExpected patch grid from 448x784 (patch=16): {expected_h}x{expected_w} = {expected_h * expected_w}")
    if n_patch == expected_h * expected_w:
        print("  -> Token count MATCHES spatial grid.")
    else:
        print("  -> Token count DOES NOT MATCH spatial grid!")

    print("\n5. Spatial Grid & Crop:")
    patches = feats[:, n_special:]
    b, n, d = patches.shape
    grid = patches.reshape(b, expected_h, expected_w, d)
    print(f"Full patch grid shape: {tuple(grid.shape)}")

    crop_h, crop_w = 14, 25
    sh = (expected_h - crop_h) // 2
    sw = (expected_w - crop_w) // 2
    crop = grid[:, sh:sh+crop_h, sw:sw+crop_w, :]
    print(f"Center crop shape ({crop_h}x{crop_w}): {tuple(crop.shape)}")

if __name__ == "__main__":
    main()
