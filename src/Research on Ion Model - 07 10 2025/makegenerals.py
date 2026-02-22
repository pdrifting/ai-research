import os
import torch
import torch.nn as nn
import random

# ----------------------------
# Configuration
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERTS_DIR = "./experts"
GENERALS_DIR = "./generals"
os.makedirs(GENERALS_DIR, exist_ok=True)

TRAIN_SAMPLES_PER_BITWIDTH = 500
MAX_EPOCHS = 1000
LOSS_THRESHOLD = 0.0001
LEARNING_RATE = 0.001

GATE_NAMES = ["not", "or", "nor", "xor", "xnor", "and", "nand"]

# ----------------------------
# Helper functions
# ----------------------------

def generate_binary_vector(bits):
    return torch.tensor([random.randint(0, 1) for _ in range(bits)], dtype=torch.float32)

def pad_to_32bit(vec):
    padded = torch.zeros(32, dtype=torch.float32)
    padded[:vec.shape[0]] = vec
    return padded

def load_expert_model(gate, bit):
    path = os.path.join(EXPERTS_DIR, f"{gate}_{bit}bit.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expert: {path}")
    expert = torch.load(path, map_location=DEVICE)
    expert.eval()
    return expert

def compute_gate_output(gate, A, B):
    # A, B are binary tensors of size `bits`
    if gate == "not":
        out = (~A.bool()).to(torch.float32)
    elif gate == "and":
        out = (A.bool() & B.bool()).to(torch.float32)
    elif gate == "or":
        out = (A.bool() | B.bool()).to(torch.float32)
    elif gate == "xor":
        out = (A.bool() ^ B.bool()).to(torch.float32)
    elif gate == "nand":
        out = (~(A.bool() & B.bool())).to(torch.float32)
    elif gate == "nor":
        out = (~(A.bool() | B.bool())).to(torch.float32)
    elif gate == "xnor":
        out = (~(A.bool() ^ B.bool())).to(torch.float32)
    else:
        raise ValueError(f"Unsupported gate: {gate}")
    return out

def generate_labeled_samples(gate, bits, expert_model, n=500):
    samples = []
    expected_input_size = bits * 2

    for _ in range(n):
        A = generate_binary_vector(bits)
        B = generate_binary_vector(bits) if gate != "not" else torch.zeros_like(A)

        a_float = A.float()
        b_float = B.float()

        inp = torch.cat([a_float, b_float]).unsqueeze(0)  # shape: [1, bits*2]
        inp = inp.to(DEVICE)  # <--- important, move input to correct device

        actual_input_size = inp.shape[1]
        if actual_input_size != expected_input_size:
            raise ValueError(f"[!] Input mismatch: Expert expects {expected_input_size}, but got {actual_input_size}")

        with torch.no_grad():
            out = expert_model(inp).squeeze().cpu()  # output back to CPU for storage

        target = compute_gate_output(gate, A, B)

        samples.append((a_float, b_float, torch.tensor([bits], dtype=torch.float32), target))

    return samples

# ----------------------------
# Generalist model
# ----------------------------

class NotExpert1Bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class MultiBitExpert(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bits * 2, bits * 8),
            nn.ReLU(),
            nn.Linear(bits * 8, bits * 4),
            nn.ReLU(),
            nn.Linear(bits * 4, bits),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class GateGeneralist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32 * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )

    def forward(self, A, B, bit_width):
        x = torch.cat([A, B, bit_width], dim=1)
        return self.net(x)

# ----------------------------
# Training loop per gate
# ----------------------------

def train_gate_generalist(gate):
    print(f"\n[+] Training generalist for {gate.upper()}")
    general_model = GateGeneralist().to(DEVICE)
    optimizer = torch.optim.Adam(general_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    all_samples = []

    bit_range = range(1, 33) if gate == "not" else range(2, 33)

    for bit in bit_range:
        try:
            print(f"    ↪ Loading {bit}-bit expert for {gate}")
            expert = load_expert_model(gate, bit)
            samples = generate_labeled_samples(gate, bit, expert, TRAIN_SAMPLES_PER_BITWIDTH)
            all_samples.extend(samples)
        except FileNotFoundError:
            print(f"[!] Missing expert {gate}_{bit}bit.pt — skipping")
            continue

    # Prepare training set
    random.shuffle(all_samples)
    inputs_A = torch.stack([s[0] for s in all_samples]).to(DEVICE)
    inputs_B = torch.stack([s[1] for s in all_samples]).to(DEVICE)
    bits     = torch.stack([s[2] for s in all_samples]).to(DEVICE)
    labels   = torch.stack([s[3] for s in all_samples]).to(DEVICE)

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        preds = general_model(inputs_A, inputs_B, bits)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        print(f"    [Epoch {epoch:03}] Loss: {loss.item():.6f}")

        if loss.item() <= LOSS_THRESHOLD:
            print(f"    [✓] Training converged — loss={loss.item():.6f}, saving...")
            torch.save(general_model, os.path.join(GENERALS_DIR, f"{gate}_general.pt"))
            return

    print(f"    [x] Max epochs reached. Final loss: {loss.item():.6f}")
    torch.save(general_model, os.path.join(GENERALS_DIR, f"{gate}_general.pt"))

# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    for gate in GATE_NAMES:
        train_gate_generalist(gate)