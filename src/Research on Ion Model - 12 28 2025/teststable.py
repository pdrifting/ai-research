import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaincc, erfc

# =================================================================
# 1. CHAOS ENGINE DEFINITION
# =================================================================

class SnakeActivation(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_features) * 0.5)
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * (torch.sin(self.alpha * x) ** 2)

class DriftLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_dim, in_dim) * 0.15)
        self.weight_sigma = nn.Parameter(torch.abs(torch.randn(out_dim, in_dim) * 0.15))
        self.snake = SnakeActivation(out_dim)
    def forward(self, x):
        w = self.weight_mu + (self.weight_sigma * torch.randn_like(self.weight_sigma) * 0.3)
        return self.snake(F.linear(x, w))

class ChaosEngine(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        self.l1 = DriftLayer(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = DriftLayer(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.ln1(self.l1(x))
        identity = x
        x = self.ln2(self.l2(x))
        x = x + identity
        return torch.sigmoid(self.out_head(x))

# =================================================================
# 2. FULL NIST BATTERY (Vectorized)
# =================================================================

class NIST:
    @staticmethod
    def monobit(bits):
        s_n = np.sum(2 * bits - 1)
        return erfc(abs(s_n) / np.sqrt(2 * len(bits)))

    @staticmethod
    def block_frequency(bits, m=128):
        n = len(bits)
        N = n // m
        proportions = [np.mean(bits[i*m:(i+1)*m]) for i in range(N)]
        chi_sq = 4 * m * np.sum((np.array(proportions) - 0.5)**2)
        return gammaincc(N/2, chi_sq/2)

    @staticmethod
    def runs(bits):
        n = len(bits)
        pi = np.mean(bits)
        if abs(pi - 0.5) >= (2/np.sqrt(n)): return 0.0
        v_obs = np.sum(bits[:-1] != bits[1:]) + 1
        return erfc(abs(v_obs - 2*n*pi*(1-pi)) / (2*np.sqrt(2*n)*pi*(1-pi)))

    @staticmethod
    def longest_run_ones(bits):
        # Simplified NIST params for 128k+ bits
        n = len(bits)
        m, k = 10000, 6
        N = n // m
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0527]
        v = np.zeros(7)
        for i in range(N):
            block = bits[i*m:(i+1)*m]
            runs = "".join(block.astype(str)).split('0')
            max_run = max([len(r) for r in runs]) if runs else 0
            if max_run <= 10: v[0] += 1
            elif max_run >= 16: v[6] += 1
            else: v[max_run-10] += 1
        chi = np.sum((v - N*np.array(pi))**2 / (N*np.array(pi)))
        return gammaincc(3, chi/2)

    @staticmethod
    def fft(bits):
        n = len(bits)
        X = np.fft.fft(2*bits - 1)
        m = np.abs(X[:n//2])
        T = np.sqrt(np.log(1/0.05) * n)
        N0 = 0.95 * n / 2
        N1 = np.sum(m < T)
        d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
        return erfc(abs(d) / np.sqrt(2))

    @staticmethod
    def rank(bits):
        M, Q = 32, 32
        num = len(bits) // (M*Q)
        mats = bits[:num*1024].reshape(num, M, Q)
        def get_rank(m):
            rank = 0
            mat = (m > 0).astype(np.int8)
            for j in range(Q):
                if rank >= M: break
                p = rank + np.argmax(mat[rank:, j])
                if mat[p, j] == 0: continue
                mat[[rank, p]] = mat[[p, rank]]
                for i in range(rank + 1, M):
                    if mat[i, j]: mat[i] ^= mat[rank]
                rank += 1
            return rank
        ranks = [get_rank(m) for m in mats]
        f32 = np.sum(np.array(ranks) == 32)
        f31 = np.sum(np.array(ranks) == 31)
        e1, e2, e3 = 0.2888*num, 0.5776*num, 0.1336*num
        chi = (f32-e1)**2/e1 + (f31-e2)**2/e2 + (num-f32-f31-e3)**2/e3
        return gammaincc(1, chi/2)

# =================================================================
# 3. SURVIVAL & RESOURCE MANAGEMENT
# =================================================================

# Global tracker for "Death Count"
# Format: {model_path: [total_tests_failed, rounds_participated]}
survival_stats = {}

def run_full_battery(bits):
    tests = {
        "Mono": NIST.monobit(bits),
        "Blck": NIST.block_frequency(bits),
        "Runs": NIST.runs(bits),
        "Long": NIST.longest_run_ones(bits),
        "FFT ": NIST.fft(bits),
        "Rank": NIST.rank(bits)
    }
    return tests

def validation_worker(gpu_id, worker_id, task_q, res_q):
    device = torch.device(f"cuda:{gpu_id}")
    torch.set_num_threads(3) # Throttle CPU
    
    while True:
        model_path = task_q.get()
        if model_path is None: break
        
        try:
            engine = ChaosEngine().to(device)
            engine.load_state_dict(torch.load(model_path, map_location=device))
            engine.eval()
            
            z = torch.randn(4096, 512, device=device)
            bits_list = []
            with torch.no_grad():
                for _ in range(5000000 // 4096): # 5M bits per check
                    probs = engine(z)
                    z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                    z = torch.roll(z, 1, dims=1)
                    bits_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())
            
            bits = np.concatenate(bits_list)
            p_values = run_full_battery(bits)
            res_q.put((model_path, p_values))
        except Exception as e:
            res_q.put((model_path, f"CRITICAL ERROR: {e}"))

# =================================================================
# 4. MAIN LOOP
# =================================================================



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    MODEL_DIR = "stable_manifolds"
    round_idx = 1
    
    while True:
        models = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        if not models: break
            
        task_q, res_q = mp.Queue(), mp.Queue()
        procs = []
        for g in range(3):
            for w in range(2):
                p = mp.Process(target=validation_worker, args=(g, w, task_q, res_q))
                p.start(); procs.append(p)
        
        for m in models: task_q.put(m)
        for _ in procs: task_q.put(None)
        
        print(f"\n{'='*80}\nROUND {round_idx} STARTING | {len(models)} Models in Queue\n{'='*80}")
        print(f"{'Model Name':<35} | Mono | Blck | Runs | Long | FFT  | Rank | Result")
        print("-" * 80)

        current_results = []
        for _ in range(len(models)):
            current_results.append(res_q.get())
        
        for p in procs: p.join()

        # Update stats and purge
        for path, pvals in current_results:
            name = os.path.basename(path)
            if isinstance(pvals, str): continue
            
            fails = sum(1 for v in pvals.values() if v < 0.01)
            
            if path not in survival_stats:
                survival_stats[path] = [0, 0] # [Total Fails, Rounds]
            
            survival_stats[path][0] += fails
            survival_stats[path][1] += 1
            
            # Display logic
            row = f"{name[:35]:<35}"
            for k in ["Mono", "Blck", "Runs", "Long", "FFT ", "Rank"]:
                v = pvals[k]
                row += f" | {'F' if v < 0.01 else '.'}   "
            
            # PURGE LOGIC: If total failures exceed 10% of total possible tests over 3 rounds
            total_possible_tests = survival_stats[path][1] * len(pvals)
            failure_rate = survival_stats[path][0] / total_possible_tests
            
            if survival_stats[path][1] >= 3 and failure_rate > 0.10:
                print(f"{row} | PURGED (Fail Rate: {failure_rate:.1%})")
                os.remove(path)
            else:
                print(f"{row} | SURVIVED (Rank: {survival_stats[path][1]})")

        round_idx += 1
        time.sleep(5)