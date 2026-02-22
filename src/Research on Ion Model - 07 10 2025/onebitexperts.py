import math
import random
import os

EXPERTS_DIR = "./custom_experts"
os.makedirs(EXPERTS_DIR, exist_ok=True)

GATES = ["not", "and", "or", "xor", "nand", "nor", "xnor", "equal", "greater_than", "less_than", "implication"]

MAX_EPOCHS = 5000

# ----------------------------
# Utils
# ----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_deriv(y):
    return y * (1 - y)

def compute_truth(gate, A, B=None):
    a = round(A)
    b = round(B) if B is not None else None
    if gate == "not":
        return [1.0 - a]
    if gate == "and":
        return [float(a and b)]
    if gate == "or":
        return [float(a or b)]
    if gate == "xor":
        return [float(a ^ b)]
    if gate == "nand":
        return [float(not (a and b))]
    if gate == "nor":
        return [float(not (a or b))]
    if gate == "xnor":
        return [float(not (a ^ b))]
    if gate == "equal":
        return [float(a == b)]
    if gate == "greater_than":
        return [float(a > b)]
    if gate == "less_than":
        return [float(a < b)]
    if gate == "implication":
        return [float((not a) or b)]
    raise ValueError(f"Unknown gate: {gate}")

# ----------------------------
# Tiny Expert Network
# ----------------------------

class ExpertNet:
    def __init__(self, input_size, hidden_size=4, init_bias=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # If init_bias True, init weights slightly positive for easier symmetry learning (XNOR)
        init_range = (0.1, 1.0) if init_bias else (-1.0, 1.0)
        self.w1 = [[random.uniform(*init_range) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.uniform(*init_range) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.h_raw = [sum(wi * xi for wi, xi in zip(w_row, x)) + bi for w_row, bi in zip(self.w1, self.b1)]
        self.h = [sigmoid(h) for h in self.h_raw]
        self.out_raw = sum(wi * hi for wi, hi in zip(self.w2, self.h)) + self.b2
        self.out = sigmoid(self.out_raw)
        return self.out

    def backward(self, x, target, lr):
        error = self.out - target[0]
        d_out = error * sigmoid_deriv(self.out)

        # Update output layer
        for i in range(self.hidden_size):
            self.w2[i] -= lr * d_out * self.h[i]
        self.b2 -= lr * d_out

        # Backprop to hidden
        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * sigmoid_deriv(self.h[i])
            for j in range(self.input_size):
                self.w1[i][j] -= lr * d_h * x[j]
            self.b1[i] -= lr * d_h

    def save(self, path):
        with open(path, "w") as f:
            f.write(repr((self.w1, self.b1, self.w2, self.b2)))

    def analyze_neurons(self):
        print("\n=== Neuron Analysis ===")
        # Rank hidden neurons by absolute weight magnitude to output
        w2_abs = [abs(w) for w in self.w2]
        ranked_neurons = sorted(enumerate(w2_abs), key=lambda x: x[1], reverse=True)

        print("Hidden neurons ranked by output weight influence (|w2|):")
        for idx, val in ranked_neurons:
            print(f" Neuron {idx}: weight magnitude = {val:.4f}")

        # Check activations across all possible inputs (0 or 1 for each input bit)
        inputs = []
        if self.input_size == 1:
            inputs = [[0.0], [1.0]]
        elif self.input_size == 2:
            inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        else:
            print("Neuron analysis currently only supports 1 or 2 input bits.")
            return

        print("\nNeuron activations per input:")
        for inp in inputs:
            self.forward(inp)
            print(f" Input: {inp}")
            for i, (pre, act) in enumerate(zip(self.h_raw, self.h)):
                print(f"  Neuron {i}: pre-activation = {pre:.4f}, activation = {act:.4f}")
            print(f"  Output raw: {self.out_raw:.4f}, Output sigmoid: {self.out:.4f}\n")

        # Suggest pruning based on low weight magnitude and low activation range
        low_weight_threshold = 0.1  # tweak as needed
        low_activation_threshold = 0.1  # neurons with activation range less than this

        print("Pruning suggestions:")
        for i, (pre, act) in enumerate(zip(self.h_raw, self.h)):
            # Measure activation range across inputs
            acts = []
            for inp in inputs:
                self.forward(inp)
                acts.append(self.h[i])
            act_range = max(acts) - min(acts)

            if w2_abs[i] < low_weight_threshold and act_range < low_activation_threshold:
                print(f" Neuron {i} is a candidate for pruning (low weight and low activation range).")

    @staticmethod
    def load(path):
        with open(path) as f:
            w1, b1, w2, b2 = eval(f.read())
        net = ExpertNet(len(w1[0]), len(w1))
        net.w1, net.b1, net.w2, net.b2 = w1, b1, w2, b2
        return net

# ----------------------------
# Training
# ----------------------------

def train_gate(gate):
    input_size = 1 if gate == "not" else 2
    path = f"{EXPERTS_DIR}/{gate}_1bit.txt"

    # Setup custom parameters for XNOR
    if gate == "xnor":
        hidden_size = 16
        lr = 0.05
        init_bias = True
    else:
        hidden_size = 8
        lr = 0.1
        init_bias = False

    if os.path.exists(path):
        model = ExpertNet.load(path)
        if verify_gate(model, gate):
            print(f"[✓] Existing {gate} expert passed verification.")
            return model

    print(f"[•] Training {gate} expert...")
    model = ExpertNet(input_size, hidden_size, init_bias)
    data = []

    for a in [0.0, 1.0]:
        for b in [0.0, 1.0] if input_size == 2 else [None]:
            x = [a] if b is None else [a, b]
            y = compute_truth(gate, a, b)
            data.append((x, y))

    for epoch in range(MAX_EPOCHS):
        random.shuffle(data)
        for x, y in data:
            model.forward(x)
            model.backward(x, y, lr)

        # More frequent verification for faster early stopping
        if epoch % 5 == 0:
            if verify_gate(model, gate, verbose=False):
                print(f"  Epoch {epoch:04} - OK")
                # Early stop as soon as verified
                break

    if verify_gate(model, gate):
        print(f"[✓] Verified {gate} — saving.")
        model.save(path)
        return model
    else:
        print(f"[x] Failed to verify {gate}")
        return None

# ----------------------------
# Verification & Multi-bit
# ----------------------------

def verify_gate(model, gate, verbose=True):
    input_size = 1 if gate == "not" else 2
    failures = 0
    for a in [0.0, 1.0]:
        for b in [0.0, 1.0] if input_size == 2 else [None]:
            x = [a] if b is None else [a, b]
            pred = round(model.forward(x))
            truth = int(compute_truth(gate, a, b)[0])
            if pred != truth:
                failures += 1
                if verbose:
                    print(f"  [!] FAIL: {gate.upper()}({a}, {b}) = {pred} != {truth}")
    return failures == 0

def multi_bit_predict(gate, A_bits, B_bits=None):
    model = ExpertNet.load(f"{EXPERTS_DIR}/{gate}_1bit.txt")
    results = []
    for i in range(len(A_bits)):
        a = A_bits[i]
        b = B_bits[i] if B_bits else None
        x = [a] if b is None else [a, b]
        out = round(model.forward(x))
        results.append(out)
    return results

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    for gate in GATES:
        train_gate(gate)

    print("\n[TEST MULTI-BIT COMPOSED]")
    A = [1.0, 0.0, 1.0, 0.0]
    B = [0.0, 1.0, 1.0, 0.0]
    for gate in GATES:
        try:
            result = multi_bit_predict(gate, A, B if gate != "not" else None)
            print(f"{gate.upper()}({A}, {B if gate != 'not' else '-'}) = {result}")
        except Exception as e:
            print(f"  [!] {gate} error: {e}")

    """
    for gate in GATES:
        model = train_gate(gate)
        if model:
            print(f"\nAnalyzing {gate} expert:")
            model.analyze_neurons()
    """