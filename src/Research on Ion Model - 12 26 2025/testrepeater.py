import numpy as np
import math
from scipy import special
from statsmodels.tsa.tsatools import lagmat

class CrossTalkCircuit:
    def __init__(self):
        # Larger Prime Lengths
        self.rA = np.random.randint(0, 2, 127) # Control Ring
        self.rB = np.random.randint(0, 2, 131) # Data Ring 1
        self.rC = np.random.randint(0, 2, 137) # Data Ring 2
        self.out_reg = 0

    def step(self):
        # 1. Always update Ring A (The Clock Master)
        self.rA = np.roll(self.rA, 1)
        self.rA[0] = self.rA[-1] ^ self.rA[7] # LFSR style
        
        # 2. Irregular Stepping: Rings B and C only move if A's bits match
        # This "Slip" is the secret to passing Linear Complexity
        if self.rA[0] == 1:
            self.rB = np.roll(self.rB, 1)
            self.rB[0] = self.rB[-1] ^ self.rB[13]
            
        if self.rA[5] == 0: # Check a different tap for C
            self.rC = np.roll(self.rC, 1)
            self.rC[0] = self.rC[-1] ^ self.rC[23]

        # 3. Non-Linear Output Function (Shrinking Generator style)
        # We XOR the current heads with a "Memory Bit" from the last state
        combined = self.rB[0] ^ self.rC[0] ^ self.out_reg
        self.out_reg = self.rB[0] # Feedback memory
        
        return combined

# --- 2. THE BIG THREE TESTS ---

def test_matrix_rank(bits, m=32, q=32):
    """NIST Binary Matrix Rank Test"""
    n = len(bits)
    num_matrices = n // (m * q)
    if num_matrices < 38: return 0.0
    
    ranks = []
    for i in range(num_matrices):
        subset = bits[i*m*q : (i+1)*m*q]
        mat = subset.reshape(m, q)
        ranks.append(np.linalg.matrix_rank(mat))
        
    f_m = ranks.count(m)
    f_m_minus_1 = ranks.count(m-1)
    f_others = num_matrices - f_m - f_m_minus_1
    
    p = [0.2888, 0.5736, 0.1376]
    v = [f_m, f_m_minus_1, f_others]
    chi_sq = sum(((v[i] - num_matrices * p[i])**2) / (num_matrices * p[i]) for i in range(3))
    return math.exp(-chi_sq / 2.0)

def test_overlapping_template(bits, template_size=9):
    """NIST Overlapping Template Test (detects periodic bias)"""
    n = len(bits)
    num_blocks = 960
    block_size = 1024
    if n < (num_blocks * block_size): return 0.0
    
    # We look for a template of all 1s as a stress test
    template = np.ones(template_size, dtype=int)
    
    counts = []
    for i in range(num_blocks):
        block = bits[i*block_size : (i+1)*block_size]
        count = 0
        for j in range(block_size - template_size + 1):
            if np.array_equal(block[j:j+template_size], template):
                count += 1
        counts.append(count)
        
    # Simplified Chi-Square for template distribution
    mu = (block_size - template_size + 1) / (2**template_size)
    var = mu * (1 - (2*template_size - 1) / (2**template_size))
    chi_sq = sum(((c - mu)**2) / var for c in counts)
    return special.gammaincc(num_blocks / 2.0, chi_sq / 2.0)

def test_linear_complexity(bits, m=500):
    """NIST Linear Complexity Test (Berlekamp-Massey proxy)"""
    n = len(bits)
    num_blocks = n // m
    if num_blocks < 500: return 0.0
    
    v = [0] * 7
    mu = m/2 + (9 + (-1)**(m+1))/36 - (m/3 + 2/9)/(2**m)
    
    for i in range(num_blocks):
        block = bits[i*m : (i+1)*m]
        # Use lagmat to build the displacement matrix for rank complexity
        L = np.linalg.matrix_rank(lagmat(block, maxlag=m//2, trim='both'))
        t = ((-1)**m) * (L - mu) + 2/9
        
        if t <= -2.5: v[0] += 1
        elif t <= -1.5: v[1] += 1
        elif t <= -0.5: v[2] += 1
        elif t <= 0.5: v[3] += 1
        elif t <= 1.5: v[4] += 1
        elif t <= 2.5: v[5] += 1
        else: v[6] += 1
        
    probs = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chi_sq = sum(((v[i] - num_blocks * probs[i])**2) / (num_blocks * probs[i]) for i in range(7))
    return special.gammaincc(3.0, chi_sq / 2.0)

# --- 3. RUNNER ---

if __name__ == "__main__":
    bit_count = 1000000 
    print(f"[*] Generating {bit_count} bits...")
    
    engine = CrossTalkCircuit()
    # Warmup
    for _ in range(1000): engine.step()
    
    stream = np.array([engine.step() for _ in range(bit_count)])
    
    print("[*] Running Big Three NIST Tests...")
    p_mat = test_matrix_rank(stream)
    p_ovl = test_overlapping_template(stream)
    p_lin = test_linear_complexity(stream)
    
    print("-" * 50)
    print(f"{'TEST':<20} | {'P-VALUE':<12} | {'RESULT'}")
    print("-" * 50)
    print(f"{'Matrix Rank':<20} | {p_mat:<12.6f} | {'PASS' if p_mat >= 0.01 else 'FAIL'}")
    print(f"{'Overlapping':<20} | {p_ovl:<12.6f} | {'PASS' if p_ovl >= 0.01 else 'FAIL'}")
    print(f"{'Linear Comp':<20} | {p_lin:<12.6f} | {'PASS' if p_lin >= 0.01 else 'FAIL'}")
    print("-" * 50)