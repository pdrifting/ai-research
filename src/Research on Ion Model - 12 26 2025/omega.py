import math
import zlib
import secrets
import numpy as np

# --- THE RAW MATH ENGINE ---
PHI = (1 + 5**0.5) / 2  # Golden Ratio
PI = math.pi

def generate_shattered_bits(seed_array, length=16384):
    """
    Uses raw modular arithmetic to reverse the NIST Monobit requirements.
    Calculates a chaotic trajectory based on a 128-element seed.
    """
    output = []
    # Collapse the 128 seed elements into a single high-precision 'starting phase'
    phase = sum(s * (PHI ** i) for i, s in enumerate(seed_array)) % PI
    
    for _ in range(length):
        # 1. Modular Bernoulli Map: h_{n+1} = (phi * h_n + pi) mod 1
        # This is a classic chaotic map that is known to pass Monobit tests
        phase = (PHI * phase + PI) % 1.0
        
        # 2. Bit-Slicing: We take the 14th decimal place to avoid 
        # floating point artifacts in the early digits
        digit = int(phase * 1e14) % 2
        
        # 3. Map to Bipolar (-1, 1) for NIST Monobit math
        output.append(1 if digit == 1 else -1)
        
    return np.array(output)

# --- THE REVERSED NIST EVALUATOR ---
def evaluate_nist_logic(data):
    # NIST Test 1: Monobit (Frequency)
    # The sum of bits should be as close to zero as possible
    monobit_sum = np.sum(data)
    n = len(data)
    s_obs = abs(monobit_sum) / math.sqrt(n)
    # P-value calculation (erfc is the standard NIST way)
    p_monobit = math.erfc(s_obs / math.sqrt(2))
    
    # NIST Test 2: Runs
    # Total transitions between 0 and 1
    # We want a transition probability of ~0.5
    v_obs = np.count_nonzero(np.diff(data)) + 1
    p_prop = np.count_nonzero(data == 1) / n
    # NIST Runs formula
    num = abs(v_obs - 2 * n * p_prop * (1 - p_prop))
    den = 2 * math.sqrt(2 * n) * p_prop * (1 - p_prop)
    p_runs = math.erfc(num / den) if den > 0 else 0
    
    # NIST Test 3: Complexity (Zlib Approximation)
    complexity = len(zlib.compress(data.tobytes())) / len(data.tobytes())
    
    return p_monobit, p_runs, complexity

# --- BRUTE FORCE REFINEMENT ---
def find_perfect_nist_seed():
    print("Starting Raw Math NIST Reversal...")
    best_p_sum = 0
    # Create an initial 128-float seed
    current_seed = [secrets.SystemRandom().uniform(-1, 1) for _ in range(128)]
    
    while True:
        # Generate bitstream
        bits = generate_shattered_bits(current_seed)
        p_mono, p_runs, comp = evaluate_nist_logic(bits)
        
        # Total fitness: We want both NIST P-values to be > 0.01 (NIST pass threshold)
        # For 'Perfect' entropy, we want them as high as possible.
        current_p_sum = p_mono + p_runs + comp
        
        if current_p_sum > best_p_sum:
            best_p_sum = current_p_sum
            print(f"!!! BREAKTHROUGH !!!")
            print(f"P-Monobit: {p_mono:.8f} | P-Runs: {p_runs:.8f} | Complexity: {comp:.8f}")
            print(f"Total Fitness: {current_p_sum:.8f}\n")
            
        # Surgical Mutation: Tweak one element of the seed using secrets
        target_idx = secrets.randbelow(128)
        mutation = (secrets.SystemRandom().uniform(-1, 1)) * 1e-10 # Very small 'Atomic' shake
        current_seed[target_idx] += mutation

if __name__ == "__main__":
    find_perfect_nist_seed()