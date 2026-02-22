import torch
import numpy as np
import os
import zlib
import multiprocessing as mp
from generator import Generator

# --- CONFIGURATION ---
GPUS = [0, 1, 2]
BASE_FOLDER = "EVOLVED_KINGS"    # Source of original unique paths
REFINED_FOLDER = "ELITE_KINGS"   # Current output and refinement source
NEW_DUMP_FOLDER = "ULTRA_KINGS"  # New directory for this stage of refinement

# Original unique starting points for reference
UNIQUE_PATHS = [
    "elite_S0.96504_1109.pt", "elite_S0.96518_3144.pt", "elite_S0.96643_3021.pt",
    "elite_S0.96555_2567.pt", "elite_S0.96685_2817.pt", "elite_S0.96796_2979.pt",
    "elite_S0.96900_2316.pt", "elite_S0.96809_3208.pt", "elite_S0.96574_3178.pt"
]

def get_fitness(model, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1024, 128).to(device)
        data = model(z).flatten().cpu().numpy()
        m1 = 1.0 - abs(np.mean(data))
        m2 = len(np.where(np.diff(np.sign(data)) != 0)[0]) / (len(data) + 1e-6) * 2.0
        m3 = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        m4 = np.mean(np.diff(data)**2) / (2 * np.max([np.var(data), 1e-6]))
        hist, _ = np.histogram(data, bins=256, range=(-1, 1))
        m5 = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        return np.mean([m1, m2, m3, m4, m5])

def find_best_state_for_stream(stream_id, base_filename):
    """
    Scans REFINED_FOLDER for the best version of this specific stream.
    If none found, falls back to the original UNIQUE_PATH in BASE_FOLDER.
    """
    search_prefix = f"STREAM_{stream_id}_S"
    best_file = None
    best_val = -1.0

    if os.path.exists(REFINED_FOLDER):
        files = [f for f in os.listdir(REFINED_FOLDER) if f.startswith(search_prefix)]
        for f in files:
            try:
                # Extract score from filename STREAM_0_S0.970123.pt
                score_str = f.split('_S')[1].replace('.pt', '')
                score = float(score_str)
                if score > best_val:
                    best_val = score
                    best_file = os.path.join(REFINED_FOLDER, f)
            except:
                continue

    if best_file:
        print(f"[*] Stream {stream_id} starting from REFINED: {os.path.basename(best_file)}")
        return best_file
    else:
        print(f"[*] Stream {stream_id} starting from BASE: {base_filename}")
        return os.path.join(BASE_FOLDER, base_filename)

def run_unique_stream(gpu_id, line_id, original_path):
    device = torch.device(f"cuda:{gpu_id}")
    
    # --- HOT SWAP: Load the absolute best found so far for this specific stream ---
    best_available_path = find_best_state_for_stream(line_id, original_path)
    
    checkpoint = torch.load(best_available_path, map_location=device)
    best_state = checkpoint['g_state']
    
    model = Generator().to(device)
    model.load_state_dict(best_state)
    best_score = get_fitness(model, device)
    
    fails = 0
    target_layer_idx = None
    burst_count = 0

    print(f"[LIVE] Stream {line_id} (G{gpu_id}) ACTIVE | Current Best: {best_score:.6f}")

    while True:
        # NON-CUMULATIVE: Fresh reload every attempt
        model.load_state_dict(best_state)
        params = list(model.parameters())
        
        if burst_count > 0 and target_layer_idx is not None:
            mode = f"BURST_{target_layer_idx}"
            shake = 0.00005 + (np.random.random() * 0.00015)
            burst_count -= 1
        elif fails < 50:
            mode = "FINE"
            shake = 0.0001 + (np.random.random() * 0.0002)
            target_layer_idx = None 
        else:
            mode = "PROBE"
            target_layer_idx = np.random.randint(0, len(params))
            shake = 10 ** np.random.uniform(-5.5, -2.5) # Slightly deeper probes

        with torch.no_grad():
            if target_layer_idx is not None:
                params[target_layer_idx].add_(torch.randn(params[target_layer_idx].size()).to(device) * shake)
            else:
                for p in model.parameters():
                    p.add_(torch.randn(p.size()).to(device) * shake)

        new_score = get_fitness(model, device)

        if new_score > best_score:
            best_score = new_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            fails = 0
            if mode == "PROBE":
                burst_count = 20 # Increased burst length for high-entropy stabilization
            
            # Save to the NEW directory for this specific refinement run
            out_name = f"{NEW_DUMP_FOLDER}/ULTRA_{line_id}_S{best_score:.6f}.pt"
            torch.save({'g_state': best_state}, out_name)
            print(f"\n[NEW ULTRA {line_id}] SUCCESS: {best_score:.6f} | Shake: {shake:.8f} | Mode: {mode}")
        else:
            fails += 1
            if fails % 250 == 0:
                print(f"L{line_id} (G{gpu_id}) | Best: {best_score:.6f} | Fails: {fails} | {mode}")
            
            if fails > 2000:
                fails = 0 # Cycle back

if __name__ == "__main__":
    os.makedirs(NEW_DUMP_FOLDER, exist_ok=True)
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i, path in enumerate(UNIQUE_PATHS):
        gpu_idx = GPUS[i % len(GPUS)]
        p = mp.Process(target=run_unique_stream, args=(gpu_idx, i, path))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()