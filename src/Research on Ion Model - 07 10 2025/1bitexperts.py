import math
import random
import os

EXPERTS_DIR = "./custom_experts"
os.makedirs(EXPERTS_DIR, exist_ok=True)

GATES = ["not", "and", "or", "xor", "nand", "nor", "xnor", "equal", "greater_than", "less_than", "implication"]
LEARNING_RATE = 0.1
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
    def __init__(self, input_size, hidden_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
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

    if os.path.exists(path):
        model = ExpertNet.load(path)
        if verify_gate(model, gate):
            print(f"[✓] Existing {gate} expert passed verification.")
            return model

    print(f"[•] Training {gate} expert...")
    model = ExpertNet(input_size)
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
            model.backward(x, y, LEARNING_RATE)

        if epoch % 500 == 0 and verify_gate(model, gate, verbose=False):
            print(f"  Epoch {epoch:04} - OK")

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
