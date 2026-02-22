import math
import zlib
import secrets
import numpy as np
import struct

def get_mantissa_shatter_bits(seed_array, length=16384):
    output = []
    # 4-body chaotic oscillators
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    
    # Initialize state from seed
    state = [sum(s * (1.618 ** i) for i, s in enumerate(seed_array[j:j+32])) % 1.0 for j in range(0, 128, 32)]
    x, y, z, w = state

    for i in range(length):
        # 1. Chaotic Iteration (Coupled Map Lattice)
        x = (math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0
        y = (math.sin(y * 13.1415 + z * 91.312) * 31415.9265) % 1.0
        z = (math.sin(z * 17.7182 + w * 18.111) * 27182.8182) % 1.0
        w = (math.sin(w * 19.1919 + x * 21.212) * 16180.3398) % 1.0

        # 2. RAW MANTISSA EXTRACTION
        # Convert the float64 to its 8-byte hex representation
        # We grab the middle bytes where the most 'chaos' lives
        bx = struct.pack('>d', x)
        by = struct.pack('>d', y)
        bz = struct.pack('>d', z)
        bw = struct.pack('>d', w)

        # 3. BITWISE INTERFERENCE (Reversing NIST Logic)
        # XOR the middle mantissa bytes of the 4 oscillators
        # This creates a 'Bit-Shatter' that is not smooth
        combined = bx[4] ^ by[5] ^ bz[6] ^ bw[7]
        
        # Extract the parity (LSP)
        res = combined % 2
        output.append(1 if res == 1 else -1)

    return np.array(output)

def evaluate_nist_brutal(data):
    n = len(data)
    p_mono = math.erfc(abs(np.sum(data)) / math.sqrt(2 * n))
    
    # Check transitions
    v_obs = np.count_nonzero(np.diff(data)) + 1
    p_prop = np.count_nonzero(data == 1) / n
    den = 2 * math.sqrt(2 * n) * p_prop * (1 - p_prop)
    p_runs = math.erfc(abs(v_obs - 2 * n * p_prop * (1 - p_prop)) / den) if den > 0 else 0

    # COMPLEXITY (Zlib) - We want this to be > 1.0
    # We use raw bytes to ensure we aren't compressing string overhead
    raw_bytes = np.packbits((data + 1) // 2).tobytes()
    comp = len(zlib.compress(raw_bytes, level=9)) / len(raw_bytes)
    
    return p_mono, p_runs, comp

def run_mantissa_refinement():
    print("--- OMEGA MANTISSA SHATTER: BIT-LEVEL REVERSAL ---")
    best_comp = 0
    seed = [secrets.SystemRandom().uniform(-1, 1) for _ in range(128)]
    
    while True:
        bits = get_mantissa_shatter_bits(seed)
        p_mono, p_runs, comp = evaluate_nist_brutal(bits)
        
        # We only care about breakthroughs where Complexity > 1.0
        # and NIST P-values are at least passing (> 0.01)
        if comp > best_comp and p_mono > 0.1 and p_runs > 0.1:
            best_comp = comp
            print(f"!!! BIT-SHATTER BREAKTHROUGH !!!")
            print(f"P-Mono: {p_mono:.6f} | P-Runs: {p_runs:.6f} | Comp Ratio: {comp:.6f}")
            print(f"Seed [0]: {seed[0]:.15f}\n")
            
        idx = secrets.randbelow(128)
        seed[idx] += (secrets.SystemRandom().uniform(-1, 1)) * 1e-12

if __name__ == "__main__":
    run_mantissa_refinement()