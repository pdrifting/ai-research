import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
from scipy.special import erfc, gammaincc

# =================================================================
# 1. CHAOS ENGINE (FOR RE-LOADING MODELS)
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
# 2. THE COMPLETED NIST CLASS
# =================================================================

import numpy as np
from scipy.special import erfc, gammaincc

class NISTFull:
    @staticmethod
    def monobit(bits):
        n = len(bits)
        # Force int64 to prevent overflow on 10M+ bits
        s_n = np.sum(bits, dtype=np.int64) * 2 - n
        s_obs = abs(s_n) / np.sqrt(n)
        return erfc(s_obs / np.sqrt(2))

    @staticmethod
    def block_freq(bits, m=128):
        n = len(bits)
        N = n // m
        if N == 0: return 0.0
        blocks = bits[:N*m].reshape(N, m)
        pi = np.mean(blocks, axis=1)
        chi_sq = 4 * m * np.sum((pi - 0.5)**2)
        return gammaincc(N/2, chi_sq/2)

    @staticmethod
    def runs(bits):
        n = len(bits)
        pi = np.mean(bits)
        if abs(pi - 0.5) >= (2 / np.sqrt(n)): return 0.0
        v_obs = np.sum(bits[:-1] != bits[1:], dtype=np.int64) + 1
        return erfc(abs(v_obs - 2*n*pi*(1-pi)) / (2*pi*(1-pi)*np.sqrt(2*n)))

    @staticmethod
    def longest_run(bits):
        n = len(bits)
        m, k = 10000, 6
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0527]
        N = n // m
        if N == 0: return 0.0
        v = np.zeros(7)
        for i in range(N):
            block = bits[i*m:(i+1)*m]
            # Efficient run-length using diffs
            padded = np.concatenate(([0], block, [0]))
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            if len(starts) == 0: 
                max_run = 0
            else:
                max_run = np.max(ends - starts)
            
            if max_run <= 10: v[0] += 1
            elif max_run >= 16: v[6] += 1
            else: v[max_run-10] += 1
        chi_sq = np.sum((v - N * np.array(pi))**2 / (N * np.array(pi)))
        return gammaincc(3, chi_sq/2)

    @staticmethod
    def rank(bits):
        M, Q = 32, 32
        N = len(bits) // (M * Q)
        if N == 0: return 0.0
        def get_rank(matrix):
            # Optimized GF2 Rank
            rank = 0
            for j in range(Q):
                if rank >= M: break
                pivot = np.argmax(matrix[rank:, j]) + rank
                if matrix[pivot, j] == 0: continue
                matrix[[rank, pivot]] = matrix[[pivot, rank]]
                for i in range(M):
                    if i != rank and matrix[i, j]:
                        matrix[i] ^= matrix[rank]
                rank += 1
            return rank
        
        sample_n = min(N, 500)
        mats = bits[:sample_n*M*Q].reshape(sample_n, M, Q).astype(np.int8)
        ranks = np.array([get_rank(mats[i]) for i in range(sample_n)])
        f32 = np.sum(ranks == 32)
        f31 = np.sum(ranks == 31)
        f_rem = sample_n - f32 - f31
        p = [0.2888, 0.5776, 0.1336]
        chi = (f32-sample_n*p[0])**2/(sample_n*p[0]) + (f31-sample_n*p[1])**2/(sample_n*p[1]) + (f_rem-sample_n*p[2])**2/(sample_n*p[2])
        return gammaincc(1, chi/2)

    @staticmethod
    def fft(bits):
        n = len(bits)
        x = 2 * bits.astype(np.int8) - 1
        f = np.abs(np.fft.fft(x)[:n//2])
        t = np.sqrt(np.log(1/0.05) * n)
        n0 = 0.95 * n / 2
        n1 = np.sum(f < t)
        d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
        return erfc(abs(d) / np.sqrt(2))

    @staticmethod
    def non_overlapping(bits):
        """Vectorized template matching to avoid hangs"""
        n = len(bits); m = 9; N = 8; M = n // N
        # Pattern 000000001 as a decimal value for fast search
        target = 1 
        w = np.zeros(N)
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            # Use bit shifting to convert windows to decimals for O(1) comparison
            windows = np.lib.stride_tricks.sliding_window_view(block, m)
            # Flatten window to decimal
            powers = 2**np.arange(m-1, -1, -1)
            dec_vals = np.sum(windows * powers, axis=1)
            # Find occurrences
            indices = np.where(dec_vals == target)[0]
            # Filter for non-overlapping
            count, last_idx = 0, -m
            for idx in indices:
                if idx >= last_idx + m:
                    count += 1
                    last_idx = idx
            w[i] = count
        mu = (M-m+1)/(2**m)
        var = M*((1/(2**m)) - (2*m-1)/(2**(2*m)))
        chi = np.sum((w-mu)**2 / var)
        return gammaincc(N/2, chi/2)

    @staticmethod
    def overlapping(bits):
        n = len(bits); m = 9; M = 1032; N = n // M
        v = np.zeros(6)
        # Using decimal conversion for speed
        target = 2**m - 1 # All ones
        powers = 2**np.arange(m-1, -1, -1)
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            windows = np.lib.stride_tricks.sliding_window_view(block, m)
            dec_vals = np.sum(windows * powers, axis=1)
            count = np.sum(dec_vals == target)
            v[min(count, 5)] += 1
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
        chi = np.sum((v - N*np.array(pi))**2 / (N*np.array(pi)))
        return gammaincc(2.5, chi/2)

    @staticmethod
    def maurer(bits):
        n = len(bits)
        L = 7 if n < 904960 else 8
        Q = 1280 if L == 7 else 2560
        K = (n // L) - Q
        if K <= 0: return 0.5
        blocks = bits[:(Q+K)*L].reshape(Q+K, L)
        powers = 2**np.arange(L-1, -1, -1)
        vals = np.sum(blocks * powers, axis=1)
        last_pos = np.zeros(2**L, dtype=np.int32)
        sum_dist = 0.0
        for i in range(Q):
            last_pos[vals[i]] = i + 1
        for i in range(Q, Q+K):
            v = vals[i]
            if last_pos[v] > 0:
                sum_dist += np.log2(i + 1 - last_pos[v])
            last_pos[v] = i + 1
        fn = sum_dist / K
        expected = {7: 6.1962507, 8: 7.1836656}[L]
        var = {7: 3.125, 8: 3.238}[L]
        c = 0.7 - 0.8/L + (4 + 32/L)*(K**(-3/L))/15
        sigma = c * np.sqrt(var/K)
        return erfc(abs(fn - expected) / (np.sqrt(2) * sigma))

    @staticmethod
    def linear_complexity(bits, M=500):
        n = len(bits); N = n // M
        if N == 0: return 0.5
        v = np.zeros(7)
        # Using bit-logic for GF2 acceleration
        for i in range(N):
            block = bits[i*M:(i+1)*M].astype(np.int8)
            L, b, c = 0, np.zeros(M, np.int8), np.zeros(M, np.int8)
            b[0] = 1; c[0] = 1; m = -1
            for j in range(M):
                d = block[j] ^ (np.sum(c[1:L+1] * block[j-L:j][::-1]) % 2) if L > 0 else block[j]
                if d != 0:
                    t = c.copy()
                    p = np.zeros(M, np.int8)
                    shift = j - m
                    p[shift:shift+M-shift] = b[:M-shift]
                    c ^= p
                    if L <= j/2:
                        L = j + 1 - L; m = j; b = t
            mean = M/2 + (9 + (-1)**(M+1))/36 - (M/3 + 2/9)/(2**M)
            T = ((-1)**M) * (L - mean) + 2/9
            if T <= -2.5: v[0] += 1
            elif T <= -1.5: v[1] += 1
            elif T <= -0.5: v[2] += 1
            elif T <= 0.5: v[3] += 1
            elif T <= 1.5: v[4] += 1
            elif T <= 2.5: v[5] += 1
            else: v[6] += 1
        pi = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        chi = np.sum((v - N*np.array(pi))**2 / (N*np.array(pi)))
        return gammaincc(3, chi/2)

    @staticmethod
    def serial(bits, m=10):
        n = len(bits)
        def get_psi(m_len):
            ext = np.concatenate((bits, bits[:m_len-1]))
            windows = np.lib.stride_tricks.sliding_window_view(ext, m_len)
            # Use unique counts on vectorized windows
            _, counts = np.unique(windows, axis=0, return_counts=True)
            return (2**m_len / n) * np.sum(counts**2, dtype=np.float64) - n
        ps1 = get_psi(m); ps2 = get_psi(m-1); ps3 = get_psi(m-2)
        return gammaincc(2**(m-2), (ps1-ps2)/2)

    @staticmethod
    def approx_entropy(bits, m=10):
        n = len(bits)
        def get_phi(m_len):
            ext = np.concatenate((bits, bits[:m_len-1]))
            windows = np.lib.stride_tricks.sliding_window_view(ext, m_len)
            _, counts = np.unique(windows, axis=0, return_counts=True)
            probs = counts / n
            return np.sum(probs * np.log(probs))
        ap_en = get_phi(m) - get_phi(m+1)
        chi = 2 * n * (np.log(2) - ap_en)
        return gammaincc(2**(m-1), chi/2)

    @staticmethod
    def cusum(bits):
        n = len(bits)
        # Prevent overflow during cumsum
        s = np.cumsum(2 * bits.astype(np.int64) - 1)
        z = np.max(np.abs(s))
        if z == 0: return 0.0
        sum_a = 0
        k_max = int((n/z - 1)/4)
        k_min = int((-n/z + 1)/4)
        for k in range(k_min, k_max + 1):
            sum_a += erfc((4*k+1)*z/np.sqrt(n)) - erfc((4*k+3)*z/np.sqrt(n))
        return 1.0 - sum_a

    @staticmethod
    def excursion(bits):
        """14. Random Excursions Test"""
        n = len(bits)
        x = 2 * bits.astype(np.int64) - 1
        s = np.concatenate(([0], np.cumsum(x), [0]))
        zero_indices = np.where(s == 0)[0]
        J = len(zero_indices) - 1
        
        if J < 500:
            return 1.0 # Insufficient cycles to fail
            
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        # pi(k) values from NIST SP 800-22 for |x|=1 and |x|>1
        pi_vals = {
            1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
            2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791] # generalized for |x|>1
        }
        
        p_values = []
        for state in states:
            v = np.zeros(6)
            for i in range(J):
                cycle = s[zero_indices[i]+1 : zero_indices[i+1]]
                count = np.sum(cycle == state)
                v[min(count, 5)] += 1
            
            p_ref = pi_vals[1] if abs(state) == 1 else pi_vals[2]
            chi_sq = np.sum(((v - J * np.array(p_ref))**2) / (J * np.array(p_ref)))
            p_values.append(gammaincc(2.5, chi_sq / 2.0))
            
        return min(p_values)

    @staticmethod
    def excursion_variant(bits):
        """15. Random Excursions Variant Test"""
        n = len(bits)
        x = 2 * bits.astype(np.int64) - 1
        s = np.cumsum(x)
        zero_indices = np.where(s == 0)[0]
        J = len(zero_indices)
        
        if J < 500:
            return 1.0
            
        # NIST covers 18 states from -9 to +9 (excluding 0)
        states = np.delete(np.arange(-9, 10), 9) # creates -9 to 8, then we shift
        states = [x for x in range(-9, 10) if x != 0]
        
        p_values = []
        for x_state in states:
            count = np.sum(s == x_state)
            # Theoretical formula: erfc(|count - J| / sqrt(2 * J * (4*|x| - 2)))
            num = abs(count - J)
            den = np.sqrt(2 * J * (4 * abs(x_state) - 2))
            p_values.append(erfc(num / den))
            
        return min(p_values)

# =================================================================
# 3. LOGGING & WORKER LOGIC
# =================================================================

def run_all_nist(bits):
    return {
        "01_Mono": NISTFull.monobit(bits),
        "02_BlkF": NISTFull.block_freq(bits),
        "03_Runs": NISTFull.runs(bits),
        "04_Long": NISTFull.longest_run(bits),
        "05_Rank": NISTFull.rank(bits),
        "06_FFT ": NISTFull.fft(bits),
        "07_NonO": NISTFull.non_overlapping(bits),
        "08_Over": NISTFull.overlapping(bits),
        "09_Maur": NISTFull.maurer(bits),
        "10_LCom": NISTFull.linear_complexity(bits),
        "11_Serl": NISTFull.serial(bits),
        "12_AEnt": NISTFull.approx_entropy(bits),
        "13_Cusm": NISTFull.cusum(bits),
        "14_Excu": NISTFull.excursion(bits),
        "15_ExVr": NISTFull.excursion_variant(bits)
    }

survival_registry = {} # {path: [fails, rounds]}

def test_worker(gpu_id, q_in, q_out):
    device = torch.device(f"cuda:{gpu_id}")
    torch.set_num_threads(3) # Keep the 3960X cool
    while True:
        path = q_in.get()
        if path is None: break
        try:
            model = ChaosEngine().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            z = torch.randn(4096, 512, device=device)
            bits_l = []
            with torch.no_grad():
                for _ in range(2500000 // 4096):
                    p = model(z); z = 0.9*z + (p-0.5)*0.2 + torch.randn_like(z)*0.05
                    z = torch.roll(z, 1, dims=1)
                    bits_l.append((p>0.5).cpu().numpy().astype(np.uint8).flatten())
            bits = np.concatenate(bits_l)
            q_out.put((path, run_all_nist(bits)))
        except Exception as e:
            q_out.put((path, str(e)))

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    round_no = 1
    while True:
        paths = [os.path.join("stable_manifolds", f) for f in os.listdir("stable_manifolds") if f.endswith(".pt")]
        if not paths: break
        q_i, q_o = mp.Queue(), mp.Queue()
        procs = [mp.Process(target=test_worker, args=(i%3, q_i, q_o)) for i in range(6)]
        for p in procs: p.start()
        for path in paths: q_i.put(path)
        for _ in procs: q_i.put(None)
        
        results = [q_o.get() for _ in range(len(paths))]
        for p in procs: p.join()

        # Reporting & CSV
        csv_name = f"nist_report_r{round_no}.csv"
        with open(csv_name, "w", newline="") as f, open("failed_models.txt", "a") as fl:
            wr = csv.writer(f)
            header = ["Model"] + [f"{i:02d}" for i in range(1,16)] + ["DeathCount"]
            wr.writerow(header)
            
            print(f"\n[ROUND {round_no}] | Models: {len(paths)}")
            for path, p_vals in results:
                name = os.path.basename(path)
                if isinstance(p_vals, str): continue
                
                if path not in survival_registry: survival_registry[path] = [0, 0]
                fails = sum(1 for v in p_vals.values() if v < 0.01)
                survival_registry[path][0] += fails
                survival_registry[path][1] += 1
                
                status_row = [name] + ["F" if v < 0.01 else "." for v in p_vals.values()]
                print(" | ".join(status_row) + f" | DC: {survival_registry[path][0]}")
                wr.writerow([name] + list(p_vals.values()) + [survival_registry[path][0]])

                # Purge Logic: Fail rate > 10% after 3 rounds
                if survival_registry[path][1] >= 3 and (survival_registry[path][0] / (15 * survival_registry[path][1])) > 0.10:
                    fl.write(f"PURGED: {name} after round {round_no}\n")
                    os.remove(path)
        
        round_no += 1
        time.sleep(10)