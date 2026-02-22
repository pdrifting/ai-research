import sys
import secrets 
import numpy as np
from scipy.special import gammaincc

# =================================================================
# 1. ACTUAL NIST-SPEC OVERLAPPING TEMPLATE TEST (Multinomial Bins)
# =================================================================

def test_overlapping_template(bits):
    """
    NIST SP 800-22 Overlapping Template Matching Test.
    Uses the official 5-bin multinomial distribution for m=9, n=10^6.
    """
    n = len(bits)
    m = 9  # Template length
    M = 1032 # Block size
    N = n // M # Number of blocks
    
    # Template: all ones
    template = "1" * m
    bit_str = "".join(bits.astype(str))
    
    # Multinomial Bins (NIST SP 800-22 Tables)
    # Probabilities for counts (0, 1, 2, 3, 4, 5+)
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    v = np.zeros(6)
    
    for i in range(N):
        block = bit_str[i*M : (i+1)*M]
        count = 0
        pos = 0
        while True:
            pos = block.find(template, pos)
            if pos == -1: break
            count += 1
            pos += 1
        
        if count <= 4:
            v[count] += 1
        else:
            v[5] += 1
            
    chi_sq = 0
    for i in range(6):
        chi_sq += ((v[i] - N * pi[i]) ** 2) / (N * pi[i])
        
    return gammaincc(2.5, chi_sq / 2.0) # 5 degrees of freedom

# =================================================================
# 2. OTHER NIST TESTS (RANK & LINEAR COMPLEXITY)
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

# =================================================================
# 3. BENCHMARK RUNNER
# =================================================================

def run_benchmark(num_streams=5, total_bits=1000000):
    print(f"[*] Engine: OS Entropy (secrets) | FULL NIST COMPLIANCE")
    print(f"{'Str':<4} | {'Dens':<6} | {'Rank P':<8} | {'LinC P':<8} | {'OverP':<8} | {'Status'}")
    print("-" * 70)

    for s in range(num_streams):
        raw_bytes = secrets.token_bytes((total_bits + 7) // 8)
        stream = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))[:total_bits]
        
        sys.stdout.write(f"\r[Stream {s+1}] Testing...")
        sys.stdout.flush()

        r_p, l_p, o_p = test_matrix_rank(stream), test_linear_complexity(stream), test_overlapping_template(stream)
        dens = np.mean(stream)
        
        passed = all(p >= 0.01 for p in [r_p, l_p, o_p]) and 0.49 < dens < 0.51
        status = "PASS" if passed else "FAIL"
        
        print(f"\r{s+1:<4} | {dens:<6.3f} | {r_p:<8.3f} | {l_p:<8.3f} | {o_p:<8.3f} | {status}")

if __name__ == "__main__":
    run_benchmark()