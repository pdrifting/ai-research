import torch
import numpy as np
import os
import zlib
import multiprocessing as mp
from generator import Generator

# --- CONFIGURATION ---
GPUS = [0, 1, 2]
ELITE_FOLDER = "EVOLVED_KINGS"
NEW_FOLDER = "ELITE_KINGS"

# The 9 most unique starting paths provided by the diversity scan
UNIQUE_PATHS = [
    "elite_S0.96504_1109.pt", 
    "elite_S0.96518_3144.pt", 
    "elite_S0.96643_3021.pt",
    "elite_S0.96555_2567.pt", 
    "elite_S0.96685_2817.pt", 
    "elite_S0.96796_2979.pt",
    "elite_S0.96900_2316.pt", 
    "elite_S0.96809_3208.pt", 
    "elite_S0.96574_3178.pt"
]

def get_fitness(model, device):
    model.eval()
    with torch.no_grad():
        # Using 1024 samples for high-precision elite verification
        z = torch.randn(1024, 128).to(device)
        data = model(z).flatten().cpu().numpy()
        
        m1 = 1.0 - abs(np.mean(data))
        m2 = len(np.where(np.diff(np.sign(data)) != 0)[0]) / (len(data) + 1e-6) * 2.0
        m3 = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        m4 = np.mean(np.diff(data)**2) / (2 * np.max([np.var(data), 1e-6]))
        hist, _ = np.histogram(data, bins=256, range=(-1, 1))
        m5 = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        
        return np.mean([m1, m2, m3, m4, m5])

def run_unique_stream(gpu_id, line_id, model_filename):
    device = torch.device(f"cuda:{gpu_id}")
    path = os.path.join(ELITE_FOLDER, model_filename)
    
    # Initialization
    checkpoint = torch.load(path, map_location=device)
    best_state = checkpoint['g_state']
    
    model = Generator().to(device)
    model.load_state_dict(best_state)
    best_score = get_fitness(model, device)
    
    fails = 0
    print(f"[*] Stream {line_id} (GPU {gpu_id}) ACTIVE | Path: {model_filename} | Base: {best_score:.6f}")

    while True:
        # NON-CUMULATIVE: Hard reset to best state on every single iteration
        model.load_state_dict(best_state)
        
        # --- RADIUS LOGIC ---
        if fails < 30:
            # User's Magic Range (0.0001 - 0.0004)
            shake = 0.0001 + (np.random.random() * 0.0003)
            mode = "SURG"
        elif fails < 120:
            # Micro-Sweep (Incremental probe starting from 0.00001)
            shake = 0.00001 + ((fails - 30) * 0.00001)
            mode = "SWEP"
        else:
            # Random Chaos Shock (Logarithmic jump)
            shake = 10 ** np.random.uniform(-4.0, -1.0)
            mode = "CHOS"

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.size()).to(device) * shake)

        new_score = get_fitness(model, device)

        if new_score > best_score:
            best_score = new_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            fails = 0
            # Save distinct stream history to prevent overwriting
            out_name = f"{NEW_FOLDER}/STREAM_{line_id}_S{best_score:.6f}.pt"
            torch.save({'g_state': best_state}, out_name)
            print(f"\n[STREAM {line_id}] UPGRADE: {best_score:.6f} | Radius: {shake:.8f} | Mode: {mode}")
        else:
            fails += 1
            if fails % 100 == 0:
                print(f"L{line_id} (G{gpu_id}) | Best: {best_score:.6f} | Stale: {fails} | {mode}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i, path in enumerate(UNIQUE_PATHS):
        gpu_idx = GPUS[i % len(GPUS)]
        p = mp.Process(target=run_unique_stream, args=(gpu_idx, i, path))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()