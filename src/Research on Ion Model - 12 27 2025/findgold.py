import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaincc

# =================================================================
# 1. THE STABILIZED CHAOS ENGINE ARCHITECTURE
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
# 2. VALIDATION UTILITIES (NIST OVERLAPPING TEMPLATE)
# =================================================================

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
# 3. GOLDEN SEARCH ENGINE (HARVESTER)
# =================================================================

def run_golden_harvest(target_count=10, bits_per_probe=200000, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("golden_models", exist_ok=True)
    
    print(f"[*] Starting Golden Search Engine on {device}")
    print(f"[*] Target: {target_count} stable manifolds | Probe size: {bits_per_probe} bits")
    print("-" * 70)
    print(f"{'Attempt':<8} | {'Density':<8} | {'OverP':<8} | {'Status'}")

    found = 0
    attempt = 0
    
    while found < target_count:
        attempt += 1
        # Initialize a new unique "Universe" (Weights)
        model = ChaosEngine().to(device)
        model.eval()
        
        # Initial Seed
        z_start = torch.randn(batch_size, 512, device=device)
        z = z_start.clone()
        stream_list = []
        
        # Probe generation
        with torch.no_grad():
            for _ in range(bits_per_probe // batch_size):
                probs = model(z)
                # Leaky feedback to stabilize path
                z = 0.95 * z + (probs - 0.5) * 0.2 + torch.randn_like(z) * 0.05
                z = torch.roll(z, 1, dims=1)
                stream_list.append((probs > 0.5).cpu().numpy().astype(np.uint8).flatten())
        
        stream = np.concatenate(stream_list)
        dens = np.mean(stream)
        
        # Filter 1: Tight Density (The "Ghost" Detector)
        if 0.498 <= dens <= 0.502:
            # Filter 2: NIST Overlapping Template (The "Entropy" Detector)
            o_p = test_overlapping_template(stream)
            
            if o_p >= 0.01:
                found += 1
                save_path = f"golden_models/manifold_{found}.pt"
                torch.save({
                    'model_state': model.state_dict(),
                    'seed_z': z_start.cpu(),
                    'density': dens,
                    'overp': o_p
                }, save_path)
                
                print(f"{attempt:<8} | {dens:<8.4f} | {o_p:<8.3f} | [SAVED {found}/{target_count}]")
            else:
                print(f"{attempt:<8} | {dens:<8.4f} | {o_p:<8.3f} | ENTROPY COLLAPSE")
        else:
            if attempt % 10 == 0:
                print(f"{attempt:<8} | {dens:<8.4f} | {'N/A':<8} | BIASED")

    print("-" * 70)
    print(f"[*] Harvest complete. 10 stable manifolds stored in /golden_models/")

if __name__ == "__main__":
    run_golden_harvest()