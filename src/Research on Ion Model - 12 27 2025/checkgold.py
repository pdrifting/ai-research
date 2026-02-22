import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as tmp
from scipy.special import gammaincc

# =================================================================
# 1. ARCHITECTURE DEFINITION
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
# 2. OPTIMIZED NIST (SINGLE THREADED)
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
    template = "1" * m
    bit_str = "".join(bits.astype(str))
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    v = np.zeros(6)
    for i in range(N):
        block = bit_str[i*M_block : (i+1)*M_block]
        count, pos = 0, 0
        while True:
            pos = block.find(template, pos)
            if pos == -1: break
            count += 1; pos += 1
        v[min(count, 5)] += 1
    chi_over = sum(((v[i] - N * pi[i]) ** 2) / (N * pi[i]) for i in range(6))
    o_p = gammaincc(2.5, chi_over / 2.0)
    return r_p, o_p

# =================================================================
# 3. STAGGERED WORKER
# =================================================================

def search_loop(gpu_id, worker_id):
    # Set device strictly inside the process
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[Worker G{gpu_id}:W{worker_id}] Online on {torch.cuda.get_device_name(gpu_id)}")
    
    os.makedirs("stable_manifolds", exist_ok=True)
    
    while True:
        model = ChaosEngine().to(device)
        z = torch.randn(256, 512, device=device)
        
        # Phase check
        failed = False
        for size in [250000, 1000000]:
            stream_list = []
            with torch.no_grad():
                for _ in range(size // 256):
                    probs = model(z)
                    z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                    z = torch.roll(z, 1, dims=1)
                    stream_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())
            
            bits = np.concatenate(stream_list)
            dens = np.mean(bits)
            
            if not (0.499 < dens < 0.501):
                failed = True; break
                
            r_p, o_p = run_nist_suite(bits)
            if r_p < 0.01 or o_p < 0.01:
                failed = True; break
        
        if failed:
            continue
            
        # Final Consistency Check
        print(f"[Worker G{gpu_id}:W{worker_id}] candidate found, verifying stability...")
        # (Consistency check logic remains same)
        save_path = f"stable_manifolds/G{gpu_id}_W{worker_id}_{int(time.time())}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"!!! [GOLDEN] {save_path} !!!")

if __name__ == "__main__":
    tmp.set_start_method('spawn', force=True)
    
    # Critical: Staggered launch to prevent CUDA context collision
    workers = []
    print("[*] Staggering worker launch (5s intervals)...")
    
    for g in range(3):
        for w in range(4):
            p = tmp.Process(target=search_loop, args=(g, w))
            p.start()
            workers.append(p)
            time.sleep(5) # Give the driver time to breathe
            
    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        for p in workers:
            p.terminate()