import torch
import numpy as np
import os
import zlib
import csv
import time
import multiprocessing as mp
from generator import Generator

# --- CONFIGURATION ---
GPUS = [0, 1, 2]
SOURCE_FOLDER = "ULTRA_KINGS" 
FINAL_DUMP = "NANO_KINGS"
METRICS_FILE = "nano_refinement_metrics.csv"

# Hardware Frontier: ensure every shake registers in float32
SAFE_MIN_SHAKE = 1.2e-8 

def get_fitness(model, device):
    model.eval()
    with torch.no_grad():
        # High resolution: 2048 samples to filter out statistical 'mirages'
        z = torch.randn(2048, 128).to(device)
        data = model(z).flatten().cpu().numpy()
        
        m1 = 1.0 - abs(np.mean(data))
        m2 = len(np.where(np.diff(np.sign(data)) != 0)[0]) / (len(data) + 1e-6) * 2.0
        m3 = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        m4 = np.mean(np.diff(data)**2) / (2 * np.max([np.var(data), 1e-6]))
        hist, _ = np.histogram(data, bins=256, range=(-1, 1))
        m5 = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        
        return np.mean([m1, m2, m3, m4, m5])

def log_metric(stream_id, layer_idx, shake, old_score, new_score):
    """Saves refinement success data to a central CSV."""
    file_exists = os.path.isfile(METRICS_FILE)
    with open(METRICS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'stream_id', 'layer_idx', 'shake', 'old_score', 'new_score', 'gain'])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            stream_id,
            layer_idx,
            f"{shake:.12f}",
            f"{old_score:.8f}",
            f"{new_score:.8f}",
            f"{new_score - old_score:.8f}"
        ])

def run_nano_refinement(gpu_id, stream_id):
    device = torch.device(f"cuda:{gpu_id}")
    
    if not os.path.exists(SOURCE_FOLDER):
        print(f"[!] Folder {SOURCE_FOLDER} not found.")
        return
        
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.startswith(f"ULTRA_{stream_id}_S")]
    if not files:
        print(f"[!] No ULTRA model for stream {stream_id}")
        return
    
    # Sort by score to get the best starting point
    best_file = sorted(files, key=lambda x: float(x.split('_S')[1].replace('.pt', '')))[-1]
    path = os.path.join(SOURCE_FOLDER, best_file)
    
    checkpoint = torch.load(path, map_location=device)
    best_state = checkpoint['g_state']
    
    model = Generator().to(device)
    model.load_state_dict(best_state)
    best_score = get_fitness(model, device)
    
    fails = 0
    print(f"[NANO] Stream {stream_id} (G{gpu_id}) STARTING at {best_score:.6f}")

    while True:
        model.load_state_dict(best_state)
        params = list(model.parameters())
        target_idx = np.random.randint(0, len(params))
        
        # --- BIT-LIMIT AWARE SCALING ---
        if fails < 3000:
            shake = 10 ** np.random.uniform(-7.92, -6.5)
            mode = "NANO"
        else:
            shake = 10 ** np.random.uniform(-6.5, -4.5)
            mode = "BRIDGE"

        if shake < SAFE_MIN_SHAKE:
            shake = SAFE_MIN_SHAKE

        with torch.no_grad():
            params[target_idx].add_(torch.randn(params[target_idx].size()).to(device) * shake)

        new_score = get_fitness(model, device)

        if new_score > best_score:
            log_metric(stream_id, target_idx, shake, best_score, new_score)
            best_score = new_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            fails = 0
            
            out_name = f"{FINAL_DUMP}/NANO_{stream_id}_S{best_score:.6f}.pt"
            torch.save({'g_state': best_state}, out_name)
            print(f"\n[STREAM {stream_id}] NANO UPGRADE: {best_score:.6f} | Shake: {shake:.9f} | Layer: {target_idx}")
        else:
            fails += 1
            if fails % 500 == 0:
                print(f"N{stream_id} (G{gpu_id}) | Best: {best_score:.6f} | Fails: {fails} | Scale: {shake:.9f}")
            
            if fails > 15000:
                print(f"[!] Stream {stream_id} saturated at {best_score:.6f}")
                fails = 0

if __name__ == "__main__":
    os.makedirs(FINAL_DUMP, exist_ok=True)
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i in range(9):
        gpu_idx = i % len(GPUS)
        p = mp.Process(target=run_nano_refinement, args=(gpu_idx, i))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()