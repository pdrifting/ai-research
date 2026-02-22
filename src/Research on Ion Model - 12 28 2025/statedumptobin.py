import torch
import numpy as np
import os

# The validated order from our checksum test
KEYS = [
    'l1.weight_mu', 'l1.weight_sigma', 'l1.snake.alpha',
    'ln1.weight', 'ln1.bias',
    'l2.weight_mu', 'l2.weight_sigma', 'l2.snake.alpha',
    'ln2.weight', 'ln2.bias',
    'out_head.weight', 'out_head.bias'
]

models_to_dump = [
    "stable_manifolds/backup/G1_W4_1766908238.pt",
    "stable_manifolds/backup/G2_W3_191.pt",
    "stable_manifolds/backup/G0_W0_643.pt"
]

os.makedirs("SNAKE_BINS", exist_ok=True)

for i, model_path in enumerate(models_to_dump, 1):
    print(f"Processing Model {i}: {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        out_bin = f"SNAKE_BINS/{i}.bin"
        
        with open(out_bin, 'wb') as f:
            for k in KEYS:
                # Ensure float32, detach from graph, move to cpu, convert to numpy
                data = state_dict[k].detach().cpu().float().numpy().flatten()
                data.tofile(f)
        
        print(f"  Successfully wrote {os.path.getsize(out_bin)} bytes to {out_bin}")
    except Exception as e:
        print(f"  FAILED to export {model_path}: {e}")

print("\nAll models exported. You are ready to run the C engine.")