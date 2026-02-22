import torch
import torch.nn as nn
import torch.optim as optim
import os

# Save path
SAVE_DIR = "experts"
os.makedirs(SAVE_DIR, exist_ok=True)

# Expert model with adaptive capacity
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

    def forward(self, x):
        return self.net(x)

# Truth functions
def bitwise_and(a, b): return a & b
def bitwise_or(a, b): return a | b
def bitwise_xor(a, b): return a ^ b
def bitwise_nand(a, b): return (~(a & b)) & 1
def bitwise_not(a, _): return (~a) & 1

# Verify correctness (50 samples)
def verify(model, truth_fn, bits, is_unary=False):
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            A = torch.randint(0, 2, (bits,), dtype=torch.bool)
            B = torch.randint(0, 2, (bits,), dtype=torch.bool) if not is_unary else torch.zeros_like(A)
            expected = truth_fn(A, B).float()
            input_tensor = torch.cat([A.float(), B.float()]).unsqueeze(0)
            output = model(input_tensor).squeeze()
            pred = (output > 0.5).float()
            if not torch.equal(pred, expected):
                return False
    return True

# Expert training routine
def train_expert(op_name, bits, truth_fn, is_unary=False):
    fname = f"{op_name}_{bits}bit.pt"
    path = os.path.join(SAVE_DIR, fname)

    # Skip if verified expert already exists
    if os.path.exists(path):
        model = MultiBitExpert(bits)
        try:
            model.load_state_dict(torch.load(path))
            if verify(model, truth_fn, bits, is_unary):
                print(f"SKIPPED: {fname} already verified.")
                return
            else:
                print(f"RETRAINING: {fname} failed verification.")
        except Exception as e:
            print(f"ERROR loading {fname}: {e}. Retraining...")

    print(f"\nTraining {op_name.upper()} expert ({bits} bits)...")
    torch.manual_seed(42)
    model = MultiBitExpert(bits)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    sample_count = 1000 #if bits < 32 else 50000

    x_data, y_data = [], []
    for _ in range(sample_count):
        A = torch.randint(0, 2, (bits,), dtype=torch.bool)
        B = torch.randint(0, 2, (bits,), dtype=torch.bool) if not is_unary else torch.zeros_like(A)
        input_tensor = torch.cat([A.float(), B.float()])
        target = truth_fn(A, B).float()
        x_data.append(input_tensor)
        y_data.append(target)

    x = torch.stack(x_data)
    y = torch.stack(y_data)

    for epoch in range(20000):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0005:
            break

    if verify(model, truth_fn, bits, is_unary):
        torch.save(model.state_dict(), path)
        print(f"SAVED: {fname} | Final Loss: {loss.item():.6f}")
    else:
        print(f"FAILED: {fname} did not pass verification.")

# Launch training for 2+ bits
if __name__ == "__main__":
    BIT_WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    OPERATIONS = [
        ("and", bitwise_and, False),
        ("or", bitwise_or, False),
        ("xor", bitwise_xor, False),
        ("nand", bitwise_nand, False),
        ("not", bitwise_not, True)
    ]

    for bits in BIT_WIDTHS:
        for name, fn, unary in OPERATIONS:
            train_expert(name, bits, fn, unary)