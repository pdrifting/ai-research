import math
import zlib
import secrets
import numpy as np

# --- IRRATIONAL ANCHORS ---
PHI = (1 + 5**0.5) / 2
PI = math.pi
E = math.e
SQRT2 = 2**0.5

def get_shattered_shuffle_bits(seed_array, length=16384):
    output = []
    # Circular buffer to store and shuffle entropy
    buffer = [0] * 256
    
    # Initialize 4-body lattice
    x = sum(s * (PHI ** i) for i, s in enumerate(seed_array[0:32])) % 1.0
    y = sum(s * (E ** i) for i, s in enumerate(seed_array[32:64])) % 1.0
    z = sum(s * (SQRT2 ** i) for i, s in enumerate(seed_array[64:96])) % 1.0
    w = sum(s * (PI ** i) for i, s in enumerate(seed_array[96:128])) % 1.0
    
    for i in range(length + 256): # Extra cycles to prime the buffer
        # 1. DYNAMIC COUPLING
        nx = math.sin(x * PI * (13.37 + w)) + math.cos(y * PHI * (7.11 + z))
        ny = math.sin(y * E * (11.05 + x)) + math.cos(z * PI * (3.14 + w))
        nz = math.sin(z * SQRT2 * (19.84 + y)) + math.cos(w * E * (2.71 + x))
        nw = math.sin(w * PI * (17.29 + z)) + math.cos(x * SQRT2 * (1.61 + y))
        
        nx, ny, nz, nw = (nx % 1.0), (ny % 1.0), (nz % 1.0), (nw % 1.0)
        
        # 2. BIT EXTRACTION
        raw_bit = 1 if (nx + nz) > (ny + nw) else -1
        
        # 3. CHAOTIC SHUFFLE (The NIST Reverse Mechanism)
        # Use oscillators to pick two indices in the buffer to swap
        idx1 = int(nx * 255)
        idx2 = int(ny * 255)
        
        # Swap and inject new entropy
        temp = buffer[idx1]
        buffer[idx1] = buffer[idx2]
        buffer[idx2] = raw_bit
        
        # Only start outputting after buffer is 'hot'
        if i >= 256:
            output.append(buffer[idx1])
        
        x, y, z, w = nx, ny, nz, nw
        
    return np.array(output)

def evaluate_nist_shatter(data):
    n = len(data)
    # P-MONOBIT
    p_mono = math.erfc(abs(np.sum(data)) / math.sqrt(2 * n))
    
    # P-RUNS
    v_obs = np.count_nonzero(np.diff(data)) + 1
    p_prop = np.count_nonzero(data == 1) / n
    num = abs(v_obs - 2 * n * p_prop * (1 - p_prop))
    den = 2 * math.sqrt(2 * n) * p_prop * (1 - p_prop)
    p_runs = math.erfc(num / den) if den > 0 else 0
    
    # SERIAL ENTROPY (4-bit patterns)
    bits = ((data + 1) // 2).astype(int)
    patterns = [tuple(bits[i:i+4]) for i in range(len(bits)-4)]
    unique, counts = np.unique(patterns, axis=0, return_counts=True)
    shannon = -np.sum((counts/len(patterns)) * np.log2(counts/len(patterns) + 1e-12))
    p_ser = shannon / 4.0
    
    # COMPLEXITY (Target > 1.0)
    comp = len(zlib.compress(data.tobytes(), level=9)) / len(data.tobytes())
    
    return p_mono, p_runs, p_ser, comp

def run_shatter_refinement():
    print("--- OMEGA SHATTER-SHUFFLE: REVERSING NIST TEMPLATES ---")
    best_fitness = 0
    seed = [secrets.SystemRandom().uniform(-1, 1) for _ in range(128)]
    
    while True:
        bits = get_shattered_shuffle_bits(seed)
        p_mono, p_runs, p_ser, comp = evaluate_nist_shatter(bits)
        
        # MASSIVE WEIGHT ON COMPLEXITY
        # We need to break the 0.03 ceiling. 
        fitness = p_mono + p_runs + (p_ser * 5.0) + (comp * 50.0)
        
        if fitness > best_fitness:
            best_fitness = fitness
            print(f"!!! ENTROPY SPIKE !!!")
            print(f"P-Mono: {p_mono:.6f} | P-Runs: {p_runs:.6f} | Serial: {p_ser:.6f} | Comp: {comp:.6f}")
            print(f"Total Fitness: {fitness:.6f}\n")
            
        idx = secrets.randbelow(128)
        seed[idx] += (secrets.SystemRandom().uniform(-1, 1)) * 1e-12

if __name__ == "__main__":
    run_shatter_refinement()