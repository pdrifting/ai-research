import numpy as np
import math
import struct
import random
from scipy import special

# --- 1. VIRTUAL LOGIC PROCESSOR ---

class LogicGate:
    def __init__(self, node_id, gate_type):
        self.node_id = node_id
        self.gate_type = gate_type
        self.input_sources = [] 
        self.current_state = random.randint(0, 1)
        self.next_state = 0
        
        # Extended Memory for Temporal Complexity
        self.register = []
        self.accumulator = 0
        
        if gate_type == 'SHIFT':
            size = random.choice([8, 16, 32, 64])
            self.register = [random.randint(0, 1) for _ in range(size)]
        elif gate_type == 'ACCUM':
            self.accumulator = random.randint(0, 1)

    def process(self, input_values):
        if not input_values: return self.current_state
        
        t = self.gate_type
        if t == 'NOT':   return 1 if input_values[0] == 0 else 0
        if t == 'AND':   return 1 if all(input_values) else 0
        if t == 'OR':    return 1 if any(input_values) else 0
        if t == 'XOR':   return sum(input_values) % 2
        if t == 'NAND':  return 0 if all(input_values) else 1
        if t == 'XNOR':  return 1 if sum(input_values) % 2 == 0 else 0
        if t == 'XAND':  return 1 if len(set(input_values)) == 1 else 0
        
        if t == 'SHIFT':
            self.register.append(input_values[0])
            return self.register.pop(0)
            
        if t == 'ACCUM':
            # Toggle internal state based on parity of inputs
            if sum(input_values) % 2 == 1:
                self.accumulator = 1 if self.accumulator == 0 else 0
            return self.accumulator
            
        return self.current_state

class DigitalCircuit:
    def __init__(self, num_nodes=20):
        self.num_nodes = num_nodes
        self.gate_types = ['NOT', 'AND', 'OR', 'XOR', 'NAND', 'XNOR', 'XAND', 'SHIFT', 'ACCUM']
        self.nodes = []
        
        for i in range(num_nodes):
            g_type = random.choice(self.gate_types)
            self.nodes.append(LogicGate(i, g_type))
            
        # Bi-directional Wiring
        for node in self.nodes:
            in_count = 1 if node.gate_type in ['SHIFT', 'NOT'] else random.randint(2, 4)
            node.input_sources = [random.randint(0, num_nodes-1) for _ in range(in_count)]
            
        self.emitter_id = random.randint(0, num_nodes-1)

    def step(self):
        # Phase 1: Snapshot
        current_states = [n.current_state for n in self.nodes]
        # Phase 2: Compute
        for node in self.nodes:
            in_vals = [current_states[src_id] for src_id in node.input_sources]
            node.next_state = node.process(in_vals)
        # Phase 3: Update
        for node in self.nodes:
            node.current_state = node.next_state
        return self.nodes[self.emitter_id].current_state

# --- 2. NIST "BIG THREE" BASELINE TESTS ---



def test_binary_matrix_rank(bits, m=32, q=32):
    n = len(bits)
    num_matrices = n // (m * q)
    if num_matrices < 38: return 0.0 # NIST minimum requirement
    
    ranks = []
    for i in range(num_matrices):
        mat = bits[i*m*q : (i+1)*m*q].reshape(m, q)
        ranks.append(np.linalg.matrix_rank(mat))
        
    v = [ranks.count(m), ranks.count(m-1), num_matrices - ranks.count(m) - ranks.count(m-1)]
    probs = [0.2888, 0.5736, 0.1376]
    chi_sq = sum(((v[i] - num_matrices * probs[i])**2) / (num_matrices * probs[i]) for i in range(3))
    return math.exp(-chi_sq / 2)

def test_non_overlapping_template(bits, template=[1]*9):
    n, m = len(bits), len(template)
    num_blocks = 8
    block_size = n // num_blocks
    
    counts = []
    for i in range(num_blocks):
        block = bits[i*block_size : (i+1)*block_size]
        count, j = 0, 0
        while j <= block_size - m:
            if np.array_equal(block[j:j+m], template):
                count += 1
                j += m
            else: j += 1
        counts.append(count)
        
    mu = (block_size - m + 1) / (2**m)
    var = block_size * ((1 / 2**m) - (2*m - 1) / (2**(2*m)))
    chi_sq = sum(((c - mu)**2) / var for c in counts)
    return special.gammaincc(num_blocks / 2, chi_sq / 2)

def test_linear_complexity(bits, m=500):
    # Uses matrix rank of displacement matrix as a proxy for Berlekamp-Massey
    from statsmodels.tsa.tsatools import lagmat
    n = len(bits)
    num_blocks = n // m
    v = [0] * 7
    mu = m/2 + (9 + (-1)**(m+1))/36 - (m/3 + 2/9)/(2**m)
    
    for i in range(num_blocks):
        block = bits[i*m : (i+1)*m]
        try:
            L = np.linalg.matrix_rank(lagmat(block, maxlag=m//2, trim='both'))
        except: L = 0
        t = ((-1)**m) * (L - mu) + 2/9
        v[min(max(int(t + 3), 0), 6)] += 1
        
    probs = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chi_sq = sum(((v[i] - num_blocks * probs[i])**2) / (num_blocks * probs[i]) for i in range(7))
    return special.gammaincc(3, chi_sq / 2)

# --- 3. FULL EXECUTION ---

def mutate_circuit(circuit):
    """Randomly modifies the topology or gates of the circuit."""
    node = random.choice(circuit.nodes)
    mutation_type = random.choice(['gate', 'wire', 'register'])
    
    if mutation_type == 'gate':
        node.gate_type = random.choice(circuit.gate_types)
    elif mutation_type == 'wire':
        node.input_sources = [random.randint(0, circuit.num_nodes-1) for _ in range(len(node.input_sources))]
    elif mutation_type == 'register' and node.gate_type == 'SHIFT':
        node.register = [random.randint(0, 1) for _ in range(random.randint(4, 128))]
    
    # Randomly move the emitter
    if random.random() < 0.1:
        circuit.emitter_id = random.randint(0, circuit.num_nodes-1)

def run_evolution(num_nodes=30, bit_len=100000): # Smaller bit_len for faster iteration
    best_circuit = DigitalCircuit(num_nodes=num_nodes)
    best_score = -1.0
    
    print(f"[*] Starting Evolution on {num_nodes} Nodes...")
    
    gen = 0
    while True:
        gen += 1
        # 1. Generate Bitstream
        stream = np.zeros(bit_len, dtype=int)
        for i in range(bit_len):
            stream[i] = best_circuit.step()
            
        # 2. Test
        p_matrix = test_binary_matrix_rank(stream)
        p_nonovlp = test_non_overlapping_template(stream)
        p_linear = test_linear_complexity(stream)
        
        current_score = p_matrix + p_nonovlp + p_linear
        
        # 3. Print Progress
        if gen % 10 == 0 or current_score > best_score:
            print(f"Gen {gen} | Best: {best_score:.6f} | Current: {current_score:.6f} | M: {p_matrix:.4f} L: {p_linear:.4f}")
            
        # 4. Success or Mutate
        if p_matrix >= 0.01 and p_nonovlp >= 0.01 and p_linear >= 0.01:
            print("!!! SUCCESS: FOUND A TOPOLOGY THAT PASSES THE BIG THREE !!!")
            break
            
        if current_score >= best_score:
            best_score = current_score
            # Keep this circuit, but mutate it for next round
            mutate_circuit(best_circuit)
        else:
            # Revert or try a totally new mutation
            mutate_circuit(best_circuit)

# --- [NIST TESTS - SAME AS BEFORE] ---
# (Include test_binary_matrix_rank, test_non_overlapping_template, test_linear_complexity)

if __name__ == "__main__":
    # We use a shorter bit_len (100k) to find a good 'shape' first
    run_evolution(num_nodes=48, bit_len=100000)