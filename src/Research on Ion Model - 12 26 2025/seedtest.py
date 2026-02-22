import torch
import torch.nn as nn
import os
from generator import Generator

# --- CONFIGURATION ---
ELITE_SEEDS = [1027, 401, 930]  # Your top performers from the scan
SAVE_DIR = "GENETIC_KINGS"

def initialize_from_seed(seed, device="cuda"):
    """
    Initializes the full Generator using the 'Royal' seeds discovered
    in the 100k scan to ensure we start on a high-entropy manifold.
    """
    torch.manual_seed(seed)
    model = Generator().to(device)
    
    # We apply a slight weight-scaling factor based on your scan's 
    # discovered 'Ideal Magnitude' (~0.046)
    with torch.no_grad():
        for p in model.parameters():
            # Standard Kaiming/Xavier initialization is usually higher;
            # we subtly nudge it toward the 'King' profile.
            p.mul_(0.95) 
            
    return model

def prepare_elite_squad():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Preparing Full-Scale Models from Elite Seeds...")
    
    for seed in ELITE_SEEDS:
        model = initialize_from_seed(seed, device)
        
        # Save as a starting checkpoint for your refinement scripts
        out_path = os.path.join(SAVE_DIR, f"SEED_{seed}_BASE.pt")
        torch.save({'g_state': model.state_dict()}, out_path)
        print(f"[+] Progenitor Saved: {out_path}")

if __name__ == "__main__":
    prepare_elite_squad()