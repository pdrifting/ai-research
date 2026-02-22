import torch
import numpy as np
import os
import zlib
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from generator import Generator

# --- CONFIG ---
GPUS = [0, 1, 2]
TOTAL_LINES = 12
GLOBAL_MIN_BAR = 0.965

def get_fitness(model, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(512, 128).to(device)
        data = model(z).flatten().cpu().numpy()
        monobit = 1.0 - abs(np.mean(data))
        runs = len(np.where(np.diff(np.sign(data)) != 0)[0]) / (len(data) + 1e-6) * 2.0
        comp = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        hist, _ = np.histogram(data, bins=256, range=(-1, 1))
        uniformity = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        return (monobit * 0.2) + (runs * 0.2) + (comp * 0.3) + (uniformity * 0.3)

def evolve_worker(line_id, gpu_id, parent_state, current_best, fails):
    device = torch.device(f"cuda:{gpu_id}")
    model = Generator().to(device)
    model.load_state_dict(parent_state)
    
    is_shock = False
    mask_ratio = 0.0
    
    if 50 < fails <= 100: mask_ratio, is_shock = 0.05, True
    elif 100 < fails <= 150: mask_ratio, is_shock = 0.10, True
    elif 150 < fails <= 200: mask_ratio, is_shock = 0.20, True
    elif 200 < fails <= 250: mask_ratio, is_shock = 0.30, True
    elif fails > 250: mask_ratio, is_shock = 0.50, True

    with torch.no_grad():
        for param in model.parameters():
            if is_shock:
                mask = torch.bernoulli(torch.full(param.size(), mask_ratio)).to(device)
                noise = torch.randn(param.size()).to(device) * 0.1 # High shock
                param.add_(noise * mask)
            else:
                # Use a larger shake if we are still climbing to the 0.965 floor
                shake = 0.0001 if current_best > 0.965 else 0.005
                param.add_(torch.randn(param.size()).to(device) * shake)
            
    new_score = get_fitness(model, device)
    
    # Logic: Accept if better than current personal best
    if new_score > current_best:
        cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return {"id": line_id, "score": new_score, "state": cpu_state, "improved": True, "ratio": mask_ratio}
    
    return {"id": line_id, "score": current_best, "state": parent_state, "improved": False, "ratio": mask_ratio}

def run_evolution():
    os.makedirs("ELITE_SAMPLES", exist_ok=True)
    line_data = []
    
    # PRE-EVOLUTION: Get true starting scores for all lines
    print("[*] Initializing bloodlines and calculating base fitness...")
    for i in range(TOTAL_LINES):
        m = Generator()
        device = torch.device(f"cuda:{i%3}")
        m.to(device)
        start_score = get_fitness(m, device)
        line_data.append({
            "state": {k:v.cpu().clone() for k,v in m.state_dict().items()},
            "best_score": start_score, 
            "fails": 0
        })

    gen = 0
    start_time = time.time()
    
    while True:
        gen += 1
        with ProcessPoolExecutor(max_workers=TOTAL_LINES) as ex:
            futures = [ex.submit(evolve_worker, i, i%3, line_data[i]["state"], line_data[i]["best_score"], line_data[i]["fails"]) for i in range(TOTAL_LINES)]
            
            print(f"\n--- GEN {gen} | Uptime: {int(time.time()-start_time)}s ---")
            print(f"{'LINE':<5} | {'BEST':<9} | {'FAILS':<5} | {'MASK':<6} | {'STATUS'}")
            print("-" * 65)
            
            for f in futures:
                res = f.result()
                idx = res["id"]
                
                if res["improved"]:
                    line_data[idx].update({"state": res["state"], "best_score": res["score"], "fails": 0})
                    status = "UPGRADE"
                    if res['score'] > GLOBAL_MIN_BAR:
                        torch.save({"g_state": res['state']}, f"ELITE_SAMPLES/L{idx}_S{res['score']:.5f}.pt")
                        status = "ELITE+"
                else:
                    line_data[idx]["fails"] += 1
                    status = "STALE"

                    if line_data[idx]["fails"] > 300:
                        status = "EXTINCT"
                        # Resetting to a fresh random state if it can't recover
                        m_new = Generator()
                        line_data[idx].update({
                            "state": {k:v.cpu() for k,v in m_new.state_dict().items()},
                            "best_score": 0.5,
                            "fails": 0
                        })

                print(f"{idx:<5} | {line_data[idx]['best_score']:.6f} | {line_data[idx]['fails']:<5} | {res['ratio']*100:>3.0f}% | {status}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_evolution()