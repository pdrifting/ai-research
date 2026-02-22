import torch
import numpy as np
import glob
import os
import shutil

# --- CONFIGURATION ---
ENTROPY_THRESHOLD = 0.70  # 70% complexity target
STD_DEV_MIN = 0.40        # Ensure data isn't "flat"
OUTPUT_DIR = "GOLD_SHARDS"
TRASH_DIR = "TRASH_SHARDS"

def calculate_shannon_normalized(tensor):
    flat_data = tensor.flatten().cpu().numpy()
    # 256 bins corresponds to 8-bit depth
    hist, _ = np.histogram(flat_data, bins=256, range=(-1, 1))
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    entropy_bits = -np.sum(probs * np.log2(probs))
    return entropy_bits / 8.0 # Returns 0.0 to 1.0

def sort_shards():
    from generator import Generator # Import your class
    
    for d in [OUTPUT_DIR, TRASH_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    files = glob.glob("shard_*.pt")
    print(f"[*] Analyzing {len(files)} shards...")

    for f in files:
        try:
            checkpoint = torch.load(f, map_location='cuda')
            netG = Generator().cuda()
            netG.load_state_dict(checkpoint['g_state'])
            netG.eval()

            with torch.no_grad():
                # Generate sample for testing
                data = netG(torch.randn(100, 128).cuda())
                
            entropy = calculate_shannon_normalized(data)
            std_dev = data.std().item()
            
            # Binary verdict
            is_gold = (entropy >= ENTROPY_THRESHOLD) and (std_dev >= STD_DEV_MIN)
            
            target = OUTPUT_DIR if is_gold else TRASH_DIR
            shutil.move(f, os.path.join(target, f))
            
            print(f"Sorted {f:30} | Ent: {entropy:.4f} | Std: {std_dev:.4f} -> {target}")

        except Exception as e:
            print(f"[!] Error processing {f}: {e}")

if __name__ == "__main__":
    sort_shards()