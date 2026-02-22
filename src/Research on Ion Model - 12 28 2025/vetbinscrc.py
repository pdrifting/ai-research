import torch
import numpy as np

def verify_raw_bin(bin_path):
    print(f"--- Checking Raw Binary: {bin_path} ---")
    
    # Define the order and sizes exactly as they are dumped
    tensors = [
        ("l1.weight_mu",    2048 * 512),
        ("l1.weight_sigma", 2048 * 512),
        ("l1.snake.alpha",  2048),
        ("ln1.weight",      2048),
        ("ln1.bias",        2048),
        ("l2.weight_mu",    2048 * 2048),
        ("l2.weight_sigma", 2048 * 2048),
        ("l2.snake.alpha",  2048),
        ("ln2.weight",      2048),
        ("ln2.bias",        2048),
        ("out_head.weight", 2048),
        ("out_head.bias",   1)
    ]

    with open(bin_path, 'rb') as f:
        for name, size in tensors:
            # Read the float32 data
            data = np.fromfile(f, dtype=np.float32, count=size)
            
            if len(data) < size:
                print(f"ERROR: Unexpected EOF for {name}. Expected {size}, got {len(data)}")
                break
                
            print(f"{name:20} | Sum: {data.sum():>12.4f} | Mean: {data.mean():>12.6f}")

def verify_bin(model_path):
    # Load the actual .pt file
    state_dict = torch.load(model_path, map_location='cpu')
    
    # These are the exact keys based on your class definitions
    keys = [
        'l1.weight_mu', 'l1.weight_sigma', 'l1.snake.alpha',
        'ln1.weight', 'ln1.bias',
        'l2.weight_mu', 'l2.weight_sigma', 'l2.snake.alpha',
        'ln2.weight', 'ln2.bias',
        'out_head.weight', 'out_head.bias'
    ]
    
    print(f"--- Fingerprints for {model_path} ---")
    for k in keys:
        if k in state_dict:
            t = state_dict[k].float()
            print(f"{k:20} | Sum: {t.sum().item():>12.4f} | Mean: {t.mean().item():>12.6f} | Shape: {list(t.shape)}")
        else:
            print(f"MISSING KEY: {k}")

# Run on one of your experts
verify_bin('stable_manifolds/backup/G1_W4_1766908238.pt')
verify_raw_bin('SNAKE_BINS/1.bin')