import torch
import numpy as np
import os

def export_to_bin(model_path, out_path):
    state_dict = torch.load(model_path, map_location='cpu')
    
    # EXPLICIT ORDER - DO NOT CHANGE
    keys = [
        'l1.weight_mu', 'l1.weight_sigma', 'l1.snake.alpha',
        'ln1.weight', 'ln1.bias',
        'l2.weight_mu', 'l2.weight_sigma', 'l2.snake.alpha',
        'ln2.weight', 'ln2.bias',
        'out_head.weight', 'out_head.bias'
    ]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'wb') as f:
        for k in keys:
            # Flatten to 1D and convert to float32
            data = state_dict[k].detach().cpu().float().numpy().flatten()
            data.tofile(f)
            print(f"Exported {k:20} | Shape: {list(state_dict[k].shape)} | Sum: {data.sum():.4f}")

# Regenerate your bins
export_to_bin('stable_manifolds/backup/G1_W4_1766908238.pt', 'SNAKE_BINS/1.bin')