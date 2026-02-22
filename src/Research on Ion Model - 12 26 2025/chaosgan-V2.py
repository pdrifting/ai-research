import torch
import numpy as np
import os
import zlib
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from generator import Generator

# --- SETTINGS ---
GPUS = [0, 1, 2]
BLOODLINES_PER_GPU = 4  
TOTAL_LINES = len(GPUS) * BLOODLINES_PER_GPU

def get_fitness(model, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(512, 128).to(device)
        data = model(z).flatten().cpu().numpy()
        
        monobit = 1.0 - abs(np.mean(data))
        runs = len(np.where(np.diff(np.sign(data)) != 0)[0]) / (len(data) + 1e-6) * 2.0
        comp = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        mssd = np.mean(np.diff(data)**2) / (2 * np.max([np.var(data), 1e-6]))
        hist, _ = np.histogram(data, bins=256, range=(-1, 1))
        chi = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        
        return np.mean([monobit, runs, comp, mssd, chi])

def evolve_worker(line_id, gpu_id, parent_state_cpu, score, fail_count):
    device = torch.device(f"cuda:{gpu_id}")
    model = Generator().to(device)
    model.load_state_dict(parent_state_cpu)
    
    # --- CHAOTIC RADIUS LOGIC ---
    if fail_count < 10:
        # Standard refined steps based on performance
        if score < 0.90:
            shake_val = 0.01
        elif score < 0.96:
            shake_val = 0.001
        else:
            shake_val = 0.0002
    else:
        # INJECT CHAOS: Logarithmic scale randomness
        # As fails climb, the potential for massive jumps increases
        # Using 10^x ensures we cover the range from 0.00001 to 1.0 effectively
        exponent = np.random.uniform(-5.0, 0.0) 
        shake_val = 10 ** exponent
        
        # Scaling the chaos: Occasionally allow even larger disruption if stuck for hundreds of gens
        if fail_count > 100:
            if np.random.random() > 0.8: # 20% chance for a "Super-Nuke"
                shake_val = np.random.uniform(0.1, 0.99)

    candidate = Generator().to(device)
    candidate.load_state_dict(parent_state_cpu)
    
    with torch.no_grad():
        for p in candidate.parameters():
            p.add_(torch.randn(p.size()).to(device) * shake_val)
    
    new_score = get_fitness(candidate, device)
    
    if new_score > score:
        cpu_state = {k: v.cpu().clone() for k, v in candidate.state_dict().items()}
        return {"line_id": line_id, "score": new_score, "state": cpu_state, "improved": True, "radius": shake_val}
    
    return {"line_id": line_id, "score": score, "state": parent_state_cpu, "improved": False, "radius": shake_val}

def run_evolution():
    os.makedirs("EVOLVED_KINGS", exist_ok=True)
    line_data = []
    
    # Initial setup
    for i in range(TOTAL_LINES):
        m = Generator()
        line_data.append({
            "current_state": {k: v.cpu().clone() for k, v in m.state_dict().items()},
            "best_state": {k: v.cpu().clone() for k, v in m.state_dict().items()},
            "score": 0.0,
            "fail_count": 0
        })

    gen = 0
    start_time = time.time()
    
    while True:
        gen += 1
        with ProcessPoolExecutor(max_workers=TOTAL_LINES) as executor:
            futures = [executor.submit(evolve_worker, i, GPUS[i%3], line_data[i]["current_state"], line_data[i]["score"], line_data[i]["fail_count"]) for i in range(TOTAL_LINES)]
            
            print(f"\n--- GEN {gen} | Runtime: {int(time.time()-start_time)}s ---")
            print(f"{'LINE':<5} | {'GPU':<3} | {'SCORE':<9} | {'RADIUS':<9} | {'STATUS'}")
            print("-" * 65)
            
            for f in futures:
                res = f.result()
                idx = res["line_id"]
                
                if res["improved"]:
                    line_data[idx].update({"current_state": res["state"], "best_state": res["state"], "score": res["score"], "fail_count": 0})
                    status = "UPGRADE"
                    if res["score"] > 0.965:
                        torch.save({"g_state": res["state"]}, f"EVOLVED_KINGS/elite_S{res['score']:.5f}_{int(time.time()-start_time)}.pt")
                else:
                    line_data[idx]["fail_count"] += 1
                    status = f"STALE({line_data[idx]['fail_count']})"
                    
                    # REVERT: Maintain a tether to the best version every 10 stales
                    if line_data[idx]["fail_count"] % 10 == 0:
                        line_data[idx]["current_state"] = line_data[idx]["best_state"]
                        status = "REVERT"

                print(f"{idx:<5} | {idx%3:<3} | {res['score']:.6f} | {res['radius']:.6f} | {status}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_evolution()