import math
import zlib
import secrets
import numpy as np

# --- IRRATIONAL ANCHORS ---
PHI = (1 + 5**0.5) / 2
PI = math.pi
E = math.e
SQRT2 = 2**0.5

def get_shattered_nist_bits(seed_array, length=16384):
    output = []
    
    # Initialize 4-body lattice
    x = sum(s * (PHI ** i) for i, s in enumerate(seed_array[0:32])) % 1.0
    y = sum(s * (E ** i) for i, s in enumerate(seed_array[32:64])) % 1.0
    z = sum(s * (SQRT2 ** i) for i, s in enumerate(seed_array[64:96])) % 1.0
    w = sum(s * (PI ** i) for i, s in enumerate(seed_array[96:128])) % 1.0
    
    for i in range(length):
        # 1. DYNAMIC COUPLING
        # We use 'w' to modulate the frequencies of x, y, and z
        # This breaks the linear resonance that caused the 1.0/1.0/0.03 score
        nx = math.sin(x * PI * (10 + w)) + math.cos(y * PHI * (5 + z))
        ny = math.sin(y * E * (10 + x)) + math.cos(z * PI * (5 + w))
        nz = math.sin(z * SQRT2 * (10 + y)) + math.cos(w * E * (5 + x))
        nw = math.sin(w * PI * (10 + z)) + math.cos(x * SQRT2 * (5 + y))
        
        # Normalize to [0, 1]
        nx, ny, nz, nw = (nx % 1.0), (ny % 1.0), (nz % 1.0), (nw % 1.0)
        
        # 2. BIT EXTRACTION (The Sieve)
        # We compare the cross-sums. This is a raw math XOR equivalent.
        res = 1 if (nx + nz) > (ny + nw) else 0
        
        # 3. STATE UPDATE
        x, y, z, w = nx, ny, nz, nw
        output.append(1 if res == 1 else -1)
        
    return np.array(output)

def evaluate_nist_advanced(data):
    n = len(data)
    
    # P-MONOBIT
    p_mono = math.erfc(abs(np.sum(data)) / math.sqrt(2 * n))
    
    # P-RUNS
    v_obs = np.count_nonzero(np.diff(data)) + 1
    p_prop = np.count_nonzero(data == 1) / n
    num = abs(v_obs - 2 * n * p_prop * (1 - p_prop))
    den = 2 * math.sqrt(2 * n) * p_prop * (1 - p_prop)
    p_runs = math.erfc(num / den) if den > 0 else 0
    
    # SERIAL ENTROPY (NIST Serial Test Reverse)
    # We check the entropy of 4-bit sequences
    bits = ((data + 1) // 2).astype(int)
    patterns = []
    for i in range(len(bits)-4):
        patterns.append(tuple(bits[i:i+4]))
    unique, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / len(patterns)
    # Target Shannon Entropy for 4 bits is 4.0
    shannon = -np.sum(probs * np.log2(probs + 1e-12))
    p_serial = shannon / 4.0
    
    # COMPLEXITY (Zlib)
    comp = len(zlib.compress(data.tobytes(), level=9)) / len(data.tobytes())
    
    return p_mono, p_runs, p_serial, comp

def run_advanced_refinement():
    print("--- OMEGA ADVANCED RAW: BREAKING RESONANCE ---")
    best_fitness = 0
    seed = [secrets.SystemRandom().uniform(-1, 1) for _ in range(128)]
    
    while True:
        bits = get_shattered_nist_bits(seed)
        p_mono, p_runs, p_ser, comp = evaluate_nist_advanced(bits)
        
        # FITNESS: We now prioritize Serial Entropy and Complexity
        # to ensure the 1.0/1.0 scores aren't just a simple repeating pattern.
        fitness = p_mono + p_runs + (p_ser * 15.0) + (comp * 10.0)
        
        if fitness > best_fitness:
            best_fitness = fitness
            print(f"!!! COMPLEXITY BREAKTHROUGH !!!")
            print(f"P-Mono: {p_mono:.6f} | P-Runs: {p_runs:.6f} | Serial: {p_ser:.6f} | Comp: {comp:.6f}")
            print(f"Total Fitness: {fitness:.6f}\n")
            
        idx = secrets.randbelow(128)
        seed[idx] += (secrets.SystemRandom().uniform(-1, 1)) * 1e-12

if __name__ == "__main__":
    run_advanced_refinement()