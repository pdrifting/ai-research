import torch
import numpy as np
import os
import math
import struct
import multiprocessing as mp
from scipy import special
from generator import Generator

# --- PRE-INITIALIZATION ---
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- NIST BATTERY (NAMED FOR TRACKING) ---

def nist_monobit(bits):
    s = 2 * bits - 1
    return special.erfc(abs(np.sum(s)) / math.sqrt(2 * len(bits)))

def nist_frequency_block(bits, m=20):
    n = len(bits)
    num_blocks = n // m
    proportions = [np.sum(bits[i*m:(i+1)*m])/m for i in range(num_blocks)]
    chi_sq = 4 * m * np.sum([(p - 0.5)**2 for p in proportions])
    return special.gammaincc(num_blocks / 2, chi_sq / 2)

def nist_runs(bits):
    pi = np.sum(bits) / len(bits)
    v_obs = np.count_nonzero(np.diff(bits)) + 1
    den = 2 * math.sqrt(2 * len(bits)) * pi * (1 - pi)
    return special.erfc(abs(v_obs - 2 * len(bits) * pi * (1 - pi)) / den) if den > 0 else 0

def nist_longest_run_ones(bits):
    n, m, k = len(bits), 10000, 6
    num_blocks = n // m
    v = [0] * 7
    for i in range(num_blocks):
        block = bits[i*m:(i+1)*m]
        max_run, current_run = 0, 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else: current_run = 0
        idx = min(max(0, max_run - 10), 6) if max_run > 10 else 0
        v[idx] += 1
    probs = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    chi_sq = np.sum([(v[i] - num_blocks * probs[i])**2 / (num_blocks * probs[i]) for i in range(7)])
    return special.gammaincc(3, chi_sq / 2)

def nist_binary_matrix_rank(bits, m=32, q=32):
    n = len(bits)
    num_matrices = n // (m * q)
    ranks = [np.linalg.matrix_rank(bits[i*m*q : (i+1)*m*q].reshape(m, q)) for i in range(num_matrices)]
    v = [ranks.count(m), ranks.count(m-1), num_matrices - ranks.count(m) - ranks.count(m-1)]
    probs = [0.2888, 0.5736, 0.1376]
    chi_sq = np.sum([(v[i] - num_matrices * probs[i])**2 / (num_matrices * probs[i]) for i in range(3)])
    return math.exp(-chi_sq / 2)

def nist_spectral(bits):
    n = len(bits)
    s = 2 * bits - 1
    f = np.fft.fft(s)
    m = np.abs(f[:n//2])
    t = math.sqrt(math.log(1/0.05) * n)
    n1 = np.count_nonzero(m < t)
    d = (n1 - 0.95 * n / 2) / math.sqrt(n * 0.95 * 0.05 / 4)
    return special.erfc(abs(d) / math.sqrt(2))

def nist_non_overlap(bits):
    n, m, template = len(bits), 9, [1]*9
    num_blocks = 8
    block_size = n // num_blocks
    counts = []
    for i in range(num_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        count, j = 0, 0
        while j <= block_size - m:
            if np.array_equal(block[j:j+m], template): count += 1; j += m
            else: j += 1
        counts.append(count)
    mu, var = (block_size - m + 1) / (2**m), block_size * ((1 / 2**m) - (2*m - 1) / (2**(2*m)))
    chi_sq = np.sum([(c - mu)**2 / var for c in counts])
    return special.gammaincc(num_blocks / 2, chi_sq / 2)

def nist_overlap(bits):
    n, m, num_blocks, block_size = len(bits), 9, 1032, 1024
    counts, template = [0] * 6, np.ones(m)
    for i in range(num_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        count = sum(1 for j in range(block_size - m + 1) if np.array_equal(block[j:j+m], template))
        counts[min(count, 5)] += 1
    probs = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    chi_sq = np.sum([(counts[i] - num_blocks * probs[i])**2 / (num_blocks * probs[i]) for i in range(6)])
    return special.gammaincc(2.5, chi_sq / 2)

def nist_linear_complexity(bits, m=500):
    from statsmodels.tsa.tsatools import lagmat
    num_blocks = len(bits) // m
    v = [0] * 7
    mu = m/2 + (9 + (-1)**(m+1))/36 - (m/3 + 2/9)/(2**m)
    for i in range(num_blocks):
        L = np.linalg.matrix_rank(lagmat(bits[i*m:(i+1)*m], maxlag=m//2, trim='both'))
        t = ((-1)**m) * (L - mu) + 2/9
        v[min(max(int(t + 3), 0), 6)] += 1
    probs = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chi_sq = np.sum([(v[i] - num_blocks * probs[i])**2 / (num_blocks * probs[i]) for i in range(7)])
    return special.gammaincc(3, chi_sq / 2)

def nist_serial(bits, m=10):
    n = len(bits)
    def psi_sq(m_len):
        padded = np.concatenate([bits, bits[:m_len-1]])
        counts = {}
        for i in range(n):
            p = tuple(padded[i:i+m_len])
            counts[p] = counts.get(p, 0) + 1
        return (2**m_len / n) * np.sum([c**2 for c in counts.values()]) - n
    return special.gammaincc(2**(m-2), (psi_sq(m) - psi_sq(m-1))/2)

def nist_approx_entropy(bits, m=10):
    n = len(bits)
    def phi(m_len):
        padded = np.concatenate([bits, bits[:m_len-1]])
        counts = {}
        for i in range(n):
            p = tuple(padded[i:i+m_len])
            counts[p] = counts.get(p, 0) + 1
        return np.sum([(c/n) * math.log(c/n + 1e-12) for c in counts.values()])
    chi_sq = 2 * n * (math.log(2) - (phi(m) - phi(m+1)))
    return special.gammaincc(2**(m-1), chi_sq / 2)

def nist_cusum(bits):
    n = len(bits)
    z = np.max(np.abs(np.cumsum(2 * bits - 1)))
    p = 1.0
    for k in range(int((-n/z+1)/4), int((n/z-1)/4)+1):
        p -= special.ndtr((4*k+1)*z/math.sqrt(n)) - special.ndtr((4*k-1)*z/math.sqrt(n))
    return p

# --- CORE LOGIC ---

TEST_NAMES = [
    "Monobit", "FreqBlk", "Runs", "LongRun", "Matrix", 
    "Spectr", "NonOvlp", "Ovlp", "Linear", "Serial", "AppEnt", "Cusum"
]

worker_gpu_sem = None

def init_worker(sem):
    global worker_gpu_sem
    worker_gpu_sem = sem
    torch.set_num_threads(1)

def worker_nist_analysis(args):
    model_path, gpu_id = args
    device = torch.device(f"cuda:{gpu_id}")
    try:
        with worker_gpu_sem:
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            model = Generator().to(device)
            model.load_state_dict(ckpt['g_state'])
            model.eval()
        
        bits, length = [], 1000000
        chaos = sum(p.mean().item() for p in model.parameters()) % 1.0 or 0.5
        with torch.no_grad():
            while len(bits) < length:
                z = torch.randn(4096, 128).to(device)
                sig = model(z).flatten().cpu().numpy()
                for v in sig:
                    chaos = (3.9999 * chaos * (1 - chaos))
                    res = (struct.pack('>f', v)[2] ^ struct.pack('>d', chaos)[5]) & 1
                    bits.append(res)
                    if len(bits) >= length: break
        
        bits = np.array(bits)
        test_funcs = [
            nist_monobit, nist_frequency_block, nist_runs, nist_longest_run_ones,
            nist_binary_matrix_rank, nist_spectral, nist_non_overlap,
            nist_overlap, nist_linear_complexity, nist_serial,
            nist_approx_entropy, nist_cusum
        ]
        
        results_map = {}
        passed_count = 0
        for name, func in zip(TEST_NAMES, test_funcs):
            p_val = func(bits)
            is_pass = p_val >= 0.01
            results_map[name] = is_pass
            if is_pass: passed_count += 1
            
        shannon = -np.sum((p := np.bincount(np.packbits(bits), minlength=256)/len(bits)*8) * np.log2(p + 1e-12))
        
        return {
            'name': os.path.basename(model_path),
            'pass': passed_count,
            'map': results_map,
            'shannon': shannon
        }
    except Exception as e:
        return {'name': os.path.basename(model_path), 'error': str(e)}

def parallel_rank_models(directory="TRASH_SHARDS", num_gpus=3):
    ctx = mp.get_context('spawn')
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
    gpu_sem = ctx.Semaphore(num_gpus * 2) 
    
    print(f"[*] Analyzing {len(files)} models. FAIL TRACKING ENABLED.")
    print("-" * 110)
    print(f"{'MODEL':<30} | {'PASS':<5} | {'FAILED TESTS'}")
    print("-" * 110)

    results = []
    global_fails = {name: 0 for name in TEST_NAMES}
    task_args = ((f, i % num_gpus) for i, f in enumerate(files))
    
    with ctx.Pool(processes=24, initializer=init_worker, initargs=(gpu_sem,)) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_nist_analysis, task_args), 1):
            if 'error' in res: continue
            
            fails = [name for name, passed in res['map'].items() if not passed]
            for f_name in fails: global_fails[f_name] += 1
            
            results.append(res)
            fail_str = ", ".join(fails) if fails else "NONE (PERFECT)"
            print(f"{res['name']:<30} | {res['pass']}/12 | {fail_str}")

    # Final Summary Table
    results.sort(key=lambda x: (x['pass'], x['shannon']), reverse=True)
    
    print("\n" + "="*50)
    print("GLOBAL FAILURE FREQUENCY (Which tests killed your models?)")
    print("-" * 50)
    for test, count in sorted(global_fails.items(), key=lambda x: x[1], reverse=True):
        print(f"{test:<15}: {count} models failed")
    
    print("\n" + "="*85)
    print(f"{'RANK':<5} | {'MODEL NAME':<35} | {'SCORE':<8} | {'SHANNON'}")
    print("-" * 85)
    for i, r in enumerate(results[:5]):
        print(f"#{i+1:<4} | {r['name']:<35} | {r['pass']}/12  | {r['shannon']:.8f}")

if __name__ == "__main__":
    parallel_rank_models()