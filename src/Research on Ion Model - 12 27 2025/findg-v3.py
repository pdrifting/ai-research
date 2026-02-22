import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as tmp
from scipy.special import gammaincc

# =================================================================
# 1. CORE ARCHITECTURE
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
        nn.init.normal_(self.out_head.weight, std=0.01)
    def forward(self, x):
        x = self.ln1(self.l1(x))
        identity = x
        x = self.ln2(self.l2(x))
        x = x + identity
        return torch.sigmoid(self.out_head(x))

# =================================================================
# 2. OPTIMIZED NIST (NUMPY VECTORIZED)
# =================================================================

def gf2_rank(matrix):
    m, n = matrix.shape
    mat = (matrix > 0).astype(np.int8)
    rank = 0
    for j in range(n):
        if rank >= m: break
        pivot = rank + np.argmax(mat[rank:, j])
        if mat[pivot, j] == 0: continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for i in range(rank + 1, m):
            if mat[i, j] == 1: mat[i] ^= mat[rank]
        rank += 1
    return rank

def run_nist_suite(bits):
    M, Q = 32, 32
    num_matrices = len(bits) // (M * Q)
    matrices = bits[:num_matrices * 1024].reshape(num_matrices, M, Q)
    ranks = np.array([gf2_rank(m) for m in matrices])
    f32, f31 = np.sum(ranks == 32), np.sum(ranks == 31)
    f_other = num_matrices - f32 - f31
    e1, e2, e3 = 0.2888*num_matrices, 0.5776*num_matrices, 0.1336*num_matrices
    chi_rank = ((f32 - e1)**2/e1) + ((f31 - e2)**2/e2) + ((f_other - e3)**2/e3)
    r_p = gammaincc(1, chi_rank/2.0)

    n, m, M_block = len(bits), 9, 1032
    N = n // M_block
    v = np.zeros(6)
    for i in range(N):
        block = bits[i*M_block : (i+1)*M_block]
        windows = np.lib.stride_tricks.sliding_window_view(block, m)
        count = np.count_nonzero(np.all(windows == 1, axis=1))
        v[min(count, 5)] += 1
    
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    chi_over = sum(((v[i] - N * pi[i]) ** 2) / (N * pi[i]) for i in range(6))
    o_p = gammaincc(2.5, chi_over / 2.0)
    return r_p, o_p

# =================================================================
# 3. HIGH-THROUGHPUT GPU SEARCHER
# =================================================================

def search_loop(gpu_id, worker_id):
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs("stable_manifolds", exist_ok=True)
    
    # HEAVY BATCHING: Increase parallelism on the GPU hardware
    # Generate more bits per step to move workload from CPU dispatch to GPU cores
    batch_size = 8192  
    
    while True:
        model = ChaosEngine().to(device)
        # Larger latent space for more entropy surface area
        z = torch.randn(batch_size, 512, device=device)
        
        valid = True
        # Stage check
        for size in [250000, 1000000, 5000000]:
            bits_list = []
            with torch.no_grad():
                # We do internal GPU loops to minimize Python overhead
                # and maximize CUDA kernel residency time
                steps = size // batch_size
                for _ in range(max(1, steps)):
                    probs = model(z)
                    z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                    z = torch.roll(z, 1, dims=1)
                    bits_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())
            
            bits = np.concatenate(bits_list)
            if not (0.499 < np.mean(bits) < 0.501): 
                valid = False; break
            
            r_p, o_p = run_nist_suite(bits)
            if r_p < 0.01 or o_p < 0.01:
                valid = False; break
        
        if not valid: continue

        # Deep Consistency
        print(f"[G{gpu_id}:W{worker_id}] Entering consistency test...")
        passed_consist = 0
        for _ in range(10):
            c_list = []
            with torch.no_grad():
                for _ in range(1000000 // batch_size):
                    probs = model(z)
                    z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                    z = torch.roll(z, 1, dims=1)
                    c_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())
            c_bits = np.concatenate(c_list)
            r_p, o_p = run_nist_suite(c_bits)
            if (0.499 < np.mean(c_bits) < 0.501) and r_p > 0.01 and o_p > 0.01:
                passed_consist += 1
            else:
                break
        
        if passed_consist == 10:
            save_name = f"stable_manifolds/G{gpu_id}_W{worker_id}_{int(time.time())}.pt"
            torch.save(model.state_dict(), save_name)
            print(f"\n[GOLDEN] {save_name} SAVED\n")

if __name__ == "__main__":
    tmp.set_start_method('spawn', force=True)
    # Increase to 6 workers per GPU (18 total) to saturate the bus
    workers = []
    for g in range(3):
        for w in range(6): 
            p = tmp.Process(target=search_loop, args=(g, w))
            p.start()
            workers.append(p)
            time.sleep(1)
            
    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        for p in workers:
            p.terminate()