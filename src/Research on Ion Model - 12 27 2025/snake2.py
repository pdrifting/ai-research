import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaincc

# =================================================================
# 1. THE STABILIZED CHAOS ENGINE
# =================================================================

class SnakeActivation(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_features) * 0.5)
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * (torch.sin(self.alpha * x) ** 2)

class DriftLayer(nn.Module):
    def __init__(self, in_dim, out_dim, drift_magnitude=0.3):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_dim, in_dim) * 0.15)
        self.weight_sigma = nn.Parameter(torch.abs(torch.randn(out_dim, in_dim) * 0.15))
        self.drift_magnitude = drift_magnitude
        self.snake = SnakeActivation(out_dim)

    def forward(self, x):
        epsilon = torch.randn_like(self.weight_sigma)
        w = self.weight_mu + (self.weight_sigma * epsilon * self.drift_magnitude)
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

    def forward(self, x, mutate=False):
        if mutate:
            x = x + torch.randn_like(x) * 0.5
        x = self.ln1(self.l1(x))
        identity = x
        x = self.ln2(self.l2(x))
        x = x + identity
        return torch.sigmoid(self.out_head(x))

# =================================================================
# 2. NIST COMPLIANT TESTS
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

def test_matrix_rank(bits):
    M, Q = 32, 32
    num_matrices = len(bits) // (M * Q)
    ranks = [gf2_rank(bits[i*1024:(i+1)*1024].reshape(M, Q)) for i in range(num_matrices)]
    ranks = np.array(ranks)
    f32, f31 = np.sum(ranks == 32), np.sum(ranks == 31)
    f_other = num_matrices - f32 - f31
    e1, e2, e3 = 0.2888*num_matrices, 0.5776*num_matrices, 0.1336*num_matrices
    chi = ((f32 - e1)**2/e1) + ((f31 - e2)**2/e2) + ((f_other - e3)**2/e3)
    return gammaincc(1, chi/2.0)

def test_linear_complexity(bits, M=500):
    def bm(block):
        n = len(block)
        b, c = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
        b[0], c[0] = 1, 1
        L, m = 0, -1
        for i in range(n):
            d = block[i]
            for j in range(1, L + 1): d ^= c[j] & block[i - j]
            if d != 0:
                t = c.copy()
                p = np.zeros(n, dtype=np.int8)
                for j in range(n - (i - m)): p[i - m + j] = b[j]
                c ^= p
                if L <= i / 2: L, m, b = i + 1 - L, i, t
        return L
    num_blocks = len(bits) // M
    sample = min(num_blocks, 200)
    complexities = np.array([bm(bits[i*M:(i+1)*M]) for i in range(sample)])
    mean_lc = M/2.0 + (9.0 + (-1)**(M+1))/36.0
    chi_sq = np.sum(((complexities - mean_lc)**2) / mean_lc)
    return gammaincc(sample/2.0, chi_sq/2.0)

def test_overlapping_template(bits):
    n, m, M = len(bits), 9, 1032
    N = n // M
    template = "1" * m
    bit_str = "".join(bits.astype(str))
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    v = np.zeros(6)
    for i in range(N):
        block, count, pos = bit_str[i*M : (i+1)*M], 0, 0
        while True:
            pos = block.find(template, pos)
            if pos == -1: break
            count += 1
            pos += 1
        v[min(count, 5)] += 1
    chi_sq = sum(((v[i] - N * pi[i]) ** 2) / (N * pi[i]) for i in range(6))
    return gammaincc(2.5, chi_sq / 2.0)

# =================================================================
# 3. RUNNER WITH LEAKY FEEDBACK
# =================================================================

def run_evaluation(num_streams=5, total_bits=1000000, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChaosEngine().to(device)
    model.eval()

    print(f"[*] Engine: {device} | Leaky Stochastic Mode")
    print(f"{'Str':<4} | {'Dens':<6} | {'Rank P':<8} | {'LinC P':<8} | {'OverP':<8} | {'Status'}")
    print("-" * 70)

    for s in range(num_streams):
        z = torch.randn(batch_size, 512, device=device)
        stream_list = []
        
        for i in range(0, total_bits, batch_size):
            mutate = (i % 100000 == 0)
            
            with torch.no_grad():
                probs = model(z, mutate=mutate)
                
                # --- THE FIX: LEAKY MEAN-REVERSION ---
                # 0.95 leak prevents 'runaway' values in the latent state
                z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                z = torch.roll(z, 1, dims=1)
                
                stream_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())

        stream = np.concatenate(stream_list)[:total_bits]
        r_p, l_p, o_p = test_matrix_rank(stream), test_linear_complexity(stream), test_overlapping_template(stream)
        dens = np.mean(stream)
        
        passed = all(p >= 0.01 for p in [r_p, l_p, o_p]) and 0.485 < dens < 0.515
        
        print(f"\r{s+1:<4} | {dens:<6.3f} | {r_p:<8.3f} | {l_p:<8.3f} | {o_p:<8.3f} | {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    run_evaluation()