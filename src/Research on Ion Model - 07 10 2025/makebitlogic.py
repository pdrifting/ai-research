import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ----------------------------
# Configuration
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERTS_DIR = "./experts"
os.makedirs(EXPERTS_DIR, exist_ok=True)

MAX_EPOCHS = 10000
LEARNING_RATE = 0.001

GATE_NAMES = [
    "not", "or", "nor", "xor", "xnor", "and", "nand",
    "equal", "greater_than", "less_than", "implication", "bitcount", "majority"
]

# ----------------------------
# Helper functions
# ----------------------------

def generate_binary_vector(bits):
    return torch.tensor([random.randint(0, 1) for _ in range(bits)], dtype=torch.float32)

def compute_truth(gate, A, B=None):
    if gate == "not":
        return (~A.bool()).float()
    elif gate == "and":
        return (A.bool() & B.bool()).float()
    elif gate == "or":
        return (A.bool() | B.bool()).float()
    elif gate == "xor":
        return (A.bool() ^ B.bool()).float()
    elif gate == "nand":
        return (~(A.bool() & B.bool())).float()
    elif gate == "nor":
        return (~(A.bool() | B.bool())).float()
    elif gate == "xnor":
        return (~(A.bool() ^ B.bool())).float()
    elif gate == "equal":
        return torch.tensor([1.0 if torch.equal(A, B) else 0.0], dtype=torch.float32)
    elif gate == "greater_than":
        return torch.tensor([1.0 if int(A.dot(2**torch.arange(len(A)))) > int(B.dot(2**torch.arange(len(B)))) else 0.0], dtype=torch.float32)
    elif gate == "less_than":
        return torch.tensor([1.0 if int(A.dot(2**torch.arange(len(A)))) < int(B.dot(2**torch.arange(len(B)))) else 0.0], dtype=torch.float32)
    elif gate == "implication":
        return ((~A.bool()) | B.bool()).float()
    elif gate == "bitcount":
        return torch.tensor([A.sum().item() / len(A)], dtype=torch.float32)  # Normalize to [0,1]
    elif gate == "majority":
        return torch.tensor([1.0 if A.sum().item() > (len(A) / 2) else 0.0], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown gate: {gate}")

# ----------------------------
# Expert model
# ----------------------------

class ExpertModel(nn.Module):
    def __init__(self, input_bits, output_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_bits, max(4, input_bits * 4)),
            nn.ReLU(),
            nn.Linear(max(4, input_bits * 4), max(2, output_bits * 2)),
            nn.ReLU(),
            nn.Linear(max(2, output_bits * 2), output_bits),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Special 1-bit NOT expert class
class Not1BitExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Verification with dump on failure
# ----------------------------

def verify(model, gate, bits):
    model.eval()
    failed_cases = []
    with torch.no_grad():
        for _ in range(16):  # increased tests for robustness
            A = generate_binary_vector(bits)
            B = generate_binary_vector(bits) if gate not in ["not", "bitcount", "majority"] else None
            inp = torch.cat([A, B]) if B is not None else A
            inp_device = inp.to(DEVICE).unsqueeze(0)
            pred = model(inp_device).squeeze()
            truth = compute_truth(gate, A, B).squeeze()
            if pred.shape != truth.shape or not torch.allclose(pred.cpu(), truth, atol=0.2):
                failed_cases.append({
                    "input_A": A.tolist(),
                    "input_B": B.tolist() if B is not None else None,
                    "predicted": pred.cpu().tolist() if pred.dim() else [pred.cpu().item()],
                    "truth": truth.tolist() if truth.dim() else [truth.item()]
                })
    if failed_cases:
        print(f"    [!] Verification failed on {len(failed_cases)} cases:")
        for case in failed_cases:
            print(f"      Input A: {case['input_A']}")
            if case["input_B"] is not None:
                print(f"      Input B: {case['input_B']}")
            print(f"      Predicted: {case['predicted']}")
            print(f"      Truth:     {case['truth']}")
        return False
    return True

# ----------------------------
# Load model safely
# ----------------------------

def load_expert_model(gate, bits):
    model_path = os.path.join(EXPERTS_DIR, f"{gate}_{bits}bit.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing expert: {model_path}")

    input_bits = bits if gate in ["not", "bitcount", "majority"] else bits * 2
    output_bits = 1 if gate in ["not", "equal", "greater_than", "less_than", "implication", "majority", "bitcount"] else bits

    # Instantiate appropriate model architecture
    if gate == "not" and bits == 1:
        model = Not1BitExpert()
    else:
        model = ExpertModel(input_bits, output_bits)

    # Load state dict safely
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)  # PyTorch 2.0+ recommended
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# Training expert
# ----------------------------

def train_expert(gate, bits):
    output_bits = 1 if gate in ["not", "equal", "greater_than", "less_than", "implication", "majority", "bitcount"] else bits
    input_bits = bits if gate in ["not", "bitcount", "majority"] else bits * 2

    if gate == "not" and bits == 1:
        model = Not1BitExpert().to(DEVICE)
    else:
        model = ExpertModel(input_bits, output_bits).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    x_train = []
    y_train = []
    for _ in range(1000):
        A = generate_binary_vector(bits)
        B = generate_binary_vector(bits) if gate not in ["not", "bitcount", "majority"] else None
        inp = torch.cat([A, B]) if B is not None else A
        out = compute_truth(gate, A, B)
        x_train.append(inp)
        y_train.append(out)

    x_train = torch.stack(x_train).to(DEVICE)
    y_train = torch.stack(y_train).to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        preds = model(x_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"    [Epoch {epoch:05}] Loss: {loss.item():.6f}")

        if verify(model, gate, bits):
            print(f"    [✓] Verified for {gate}_{bits}bit — saving")
            # Save only the state dict for safer loading later
            torch.save(model.state_dict(), os.path.join(EXPERTS_DIR, f"{gate}_{bits}bit.pt"))
            return

    print(f"    [x] Failed to converge for {gate}_{bits}bit")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    for gate in GATE_NAMES:
        print(f"\n=== Training Experts for {gate.upper()} ===")
        start_bit = 1 if gate in ["not", "bitcount", "majority"] else 2
        for bits in range(start_bit, 33):
            model_path = os.path.join(EXPERTS_DIR, f"{gate}_{bits}bit.pt")
            if os.path.exists(model_path):
                try:
                    existing_model = load_expert_model(gate, bits)
                    if verify(existing_model, gate, bits):
                        print(f"    [✓] Existing {gate}_{bits}bit model verified — skipping training")
                        continue
                    else:
                        print(f"    [!] Existing {gate}_{bits}bit failed verification. Retraining...")
                except Exception as e:
                    print(f"    [!] Error loading {gate}_{bits}bit: {e}. Retraining...")

            try:
                train_expert(gate, bits)
            except Exception as e:
                print(f"    [!] Error training {gate}_{bits}bit: {e}")
