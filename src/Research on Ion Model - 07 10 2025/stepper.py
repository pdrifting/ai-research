import os
import sys
import math
import random
import msvcrt

EXPERTS_DIR = "./experts_sliced"
os.makedirs(EXPERTS_DIR, exist_ok=True)

MAX_EPOCHS = 10000
LEARNING_RATE = 0.1

GATE_NAMES = [
    "not", "or", "nor", "xor", "xnor", "and", "nand",
    "equal", "greater_than", "less_than", "implication"
]

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
        return [float(a == b)]
    if gate == "implication":
        return [float((not a) or b)]
    if gate == "equal":
        return [float(a == b)]
    if gate == "greater_than":
        return [float(a > b)]
    if gate == "less_than":
        return [float(a < b)]
    raise ValueError(f"Unknown gate: {gate}")

class ExpertNet:
    def __init__(self, input_size, hidden_size=8):
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
        for i in range(self.hidden_size):
            self.w2[i] -= lr * d_out * self.h[i]
        self.b2 -= lr * d_out
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

def clear_terminal():
    os.system('cls')

def color(val, old):
    if val > old: return f"\033[92m{val:.4f}\033[0m"  # green up
    if val < old: return f"\033[91m{val:.4f}\033[0m"  # red down
    return f"\033[90m{val:.4f}\033[0m"                # gray no change

def visualize_weights(old, new, epoch, gate):
    clear_terminal()
    print(f"=== {gate.upper()} | Epoch {epoch} ===")
    print("Weights input -> hidden (w1):")
    for i, (w_row_old, w_row_new) in enumerate(zip(old[0], new[0])):
        diffs = " ".join(color(nv, ov) for nv, ov in zip(w_row_new, w_row_old))
        print(f"  Neuron {i}: {diffs}")
    print("Biases (b1):")
    print(" ".join(color(n, o) for n, o in zip(new[1], old[1])))
    print("Weights hidden -> output (w2):")
    print(" ".join(color(n, o) for n, o in zip(new[2], old[2])))
    print("Output bias (b2):")
    print(color(new[3], old[3]))

def verify_gate(model, gate):
    failures = 0
    for a in [0.0, 1.0]:
        for b in [0.0, 1.0] if model.input_size == 2 else [None]:
            x = [a] if b is None else [a, b]
            pred = round(model.forward(x))
            truth = round(compute_truth(gate, a, b)[0])
            if pred != truth:
                print(f"      FAIL: {gate.upper()}({a}, {b if b is not None else '-'}) = {pred} != {truth}")
                failures += 1
    return failures == 0

def copy_weights(model):
    return ([row[:] for row in model.w1], model.b1[:], model.w2[:], model.b2)

def train_gate(gate):
    input_size = 1 if gate == "not" else 2
    path = os.path.join(EXPERTS_DIR, f"{gate}_1bit.txt")

    model = ExpertNet.load(path) if os.path.exists(path) else ExpertNet(input_size)
    data = [([a] if b is None else [a, b], compute_truth(gate, a, b))
            for a in [0.0, 1.0] for b in [0.0, 1.0] if input_size == 2 or b is None]

    epoch = 0
    prev_weights = copy_weights(model)
    history = [prev_weights]

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"\xe0":
                key2 = msvcrt.getch()
                if key2 == b"M":  # Right arrow - forward step
                    if epoch + 1 < len(history):
                        epoch += 1
                        visualize_weights(history[epoch - 1], history[epoch], epoch, gate)
                    else:
                        # Generate next epoch
                        random.shuffle(data)
                        for x, y in data:
                            model.forward(x)
                            model.backward(x, y, LEARNING_RATE)
                        epoch += 1
                        new_weights = copy_weights(model)
                        history.append(new_weights)
                        visualize_weights(history[epoch - 1], new_weights, epoch, gate)
                        if verify_gate(model, gate):
                            print(f"\n[✓] Verified {gate} — saving\n")
                            model.save(path)
                            return
                elif key2 == b"K":  # Left arrow - backward step
                    if epoch > 0:
                        epoch -= 1
                        visualize_weights(history[epoch], history[epoch], epoch, gate)
            elif key.lower() == b'r':
                # Refresh current
                visualize_weights(history[epoch], history[epoch], epoch, gate)
            elif key.lower() == b's':
                # Save and move on
                model.save(path)
                print(f"[SAVED] {gate} expert model at epoch {epoch}")
                return
            elif key.lower() == b'j':
                # Jump forward generate next epoch
                if epoch + 1 == len(history):
                    random.shuffle(data)
                    for x, y in data:
                        model.forward(x)
                        model.backward(x, y, LEARNING_RATE)
                    epoch += 1
                    new_weights = copy_weights(model)
                    history.append(new_weights)
                    visualize_weights(history[epoch - 1], new_weights, epoch, gate)
                    if verify_gate(model, gate):
                        print(f"\n[✓] Verified {gate} — saving\n")
                        model.save(path)
                        return
                else:
                    epoch += 1
                    visualize_weights(history[epoch - 1], history[epoch], epoch, gate)
        else:
            pass  # no key pressed, do nothing

if __name__ == "__main__":
    for gate in GATE_NAMES:
        print(f"\n=== Training Bitwise Expert for {gate.upper()} ===")
        train_gate(gate)
