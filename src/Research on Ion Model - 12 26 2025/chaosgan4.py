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
REVERT_STEP = 10  # Revert to best state every 10 fails

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
    
    # --- LOGIC: REVERTS THEN EROSION ---
    # Fails 0-50: Normal evolution with 5 reverts (every 10 fails)
    # Fails 51+: Start Erosion phases
    
    if current_best < 0.90:
        shake, mode_label = 0.01, "WARP"
    elif current_best < 0.93:
        shake, mode_label = 0.005, "CRUISE"
    else:
        if fails <= 50:
            # High-Level Fine Tuning with Reverts
            if fails < 5: shake, mode_label = 0.0001, "MICRO"
            elif fails < 10: shake, mode_label = 0.00025, "MICRO2"
            elif fails < 15: shake, mode_label = 0.0005, "SURGERY"
            elif fails < 20: shake, mode_label = 0.001, "STEADY"
            elif fails < 30: shake, mode_label = 0.0025, "STEADY25"
            elif fails < 35: shake, mode_label = 0.005, "STEADY5"
            elif fails < 45: shake, mode_label = 0.0075, "STEADY75"
            elif fails <= 50: shake, mode_label = 0.01, "VOLATILE"
            else: shake, mode_label = 0.015, "VOLATILE15"
        else:
            # 5 Reverts failed to break plateau -> Enter Erosion
            is_shock = True
            mode_label = "EROSION"
            if fails <= 100: mask_ratio, shake, mode_label = 0.05, 0.0001, "MICRO (EROSION)"
            elif fails <= 150: mask_ratio, shake, mode_label = 0.10, 0.0001, "MICRO (EROSION)"
            elif fails <= 200: mask_ratio, shake, mode_label = 0.20, 0.0001, "MICRO (EROSION)"
            elif fails <= 250: mask_ratio, shake, mode_label = 0.30, 0.0001, "MICRO (EROSION)"
            elif fails <= 300: mask_ratio, shake, mode_label = 0.50, 0.0001, "MICRO (EROSION)"
            else: mask_ratio, shake, mode_label, fails = 0.0, 0.0001, "MICRO (EROSION RESET)", 0

    with torch.no_grad():
        for param in model.parameters():
            if is_shock:
                mask = torch.bernoulli(torch.full(param.size(), mask_ratio)).to(device)
                noise = torch.randn(param.size()).to(device) * 0.1 
                param.add_(noise * mask)
            else:
                param.add_(torch.randn(param.size()).to(device) * shake)
            
    new_score = get_fitness(model, device)
    
    if new_score > current_best:
        cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return {"id": line_id, "score": new_score, "state": cpu_state, "improved": True, "label": mode_label}
    
    return {"id": line_id, "score": current_best, "state": parent_state, "improved": False, "label": mode_label}

def run_evolution():
    os.makedirs("ELITE_SAMPLES", exist_ok=True)
    line_data = []
    
    for i in range(TOTAL_LINES):
        m = Generator()
        device = torch.device(f"cuda:{i%3}")
        m.to(device)
        start_score = get_fitness(m, device)
        line_data.append({
            "current_state": {k:v.cpu().clone() for k,v in m.state_dict().items()},
            "best_state": {k:v.cpu().clone() for k,v in m.state_dict().items()},
            "best_score": start_score, 
            "fails": 0
        })

    gen, start_time = 0, time.time()
    
    while True:
        gen += 1
        with ProcessPoolExecutor(max_workers=TOTAL_LINES) as ex:
            futures = [ex.submit(evolve_worker, i, i%3, line_data[i]["current_state"], line_data[i]["best_score"], line_data[i]["fails"]) for i in range(TOTAL_LINES)]
            
            print(f"\n--- GEN {gen} | Uptime: {int(time.time()-start_time)}s ---")
            print(f"{'LINE':<5} | {'BEST':<9} | {'FAILS':<5} | {'MODE'}")
            print("-" * 50)
            
            for f in futures:
                res = f.result()
                idx = res["id"]
                
                if res["improved"]:
                    line_data[idx].update({
                        "current_state": res["state"], 
                        "best_state": res["state"], 
                        "best_score": res["score"], 
                        "fails": 0
                    })
                    print(f"{idx:<5} | {res['score']:.6f} | 0     | {res['label']} WIN")
                    if res['score'] > GLOBAL_MIN_BAR:
                        torch.save({"g_state": res['state']}, f"ELITE_SAMPLES/L{idx}_S{res['score']:.5f}.pt")
                else:
                    line_data[idx]["fails"] += 1
                    f_count = line_data[idx]["fails"]
                    
                    # REVERT MECHANIC: Reset to best known state every 10 fails
                    if f_count % REVERT_STEP == 0 and f_count <= 50:
                        line_data[idx]["current_state"] = line_data[idx]["best_state"]
                        status_str = f"{res['label']} REVERT"
                    else:
                        status_str = res['label']

                    if f_count > 300:
                        m_new = Generator()
                        line_data[idx].update({
                            "current_state": {k:v.cpu() for k,v in m_new.state_dict().items()},
                            "best_state": {k:v.cpu() for k,v in m_new.state_dict().items()},
                            "best_score": 0.5, "fails": 0
                        })
                        print(f"{idx:<5} | RESET     | 0     | EXTINCT")
                    else:
                        print(f"{idx:<5} | {line_data[idx]['best_score']:.6f} | {f_count:<5} | {status_str}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_evolution()