import torch
import os
from generator import Generator

def probe_precision_limit(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    model = Generator().to(device)
    model.load_state_dict(checkpoint['g_state'])
    
    # Target a representative weight from a deep layer
    # (Deep layers often have smaller weights and hit the wall sooner)
    params = list(model.parameters())
    test_param = params[len(params)//2] 
    original_val = test_param[0].flatten()[0].item()
    
    print(f"--- PRECISION PROBE: {os.path.basename(model_path)} ---")
    print(f"Target Weight Value: {original_val:.15f}\n")
    print(f"{'Shake Scale':<15} | {'Delta Applied':<20} | {'Hardware Success'}")
    print("-" * 60)

    # Test scales from 1e-4 down to 1e-12
    scales = [10**-i for i in range(4, 13)]
    
    for s in scales:
        # Create a fresh tensor to avoid accumulation errors during the probe
        w_tensor = torch.tensor([original_val], dtype=torch.float32, device=device)
        shake = torch.tensor([s], dtype=torch.float32, device=device)
        
        new_val_tensor = w_tensor + shake
        new_val = new_val_tensor.item()
        
        success = "YES" if original_val != new_val else "FAILED (Wall Hit)"
        actual_diff = new_val - original_val
        
        print(f"10^{len(str(s))-2:<10} | {actual_diff:<20.15f} | {success}")

    print("\nConclusion:")
    if original_val != (torch.tensor([original_val], dtype=torch.float32) + 1e-8).item():
        print("-> Your hardware is still resolving 1e-8. You can proceed with NANO.")
    else:
        print("-> DEAD END: You have hit the float32 limit. Mutations smaller than this are being rounded to zero.")

# Run on one of your ULTRA models
# Replace with an actual filename from your folder
probe_precision_limit("ULTRA_KINGS/ULTRA_0_S0.968110.pt")