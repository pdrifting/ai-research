import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from torch.utils.data import TensorDataset, DataLoader

SAVE_DIR = "experts"
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scalable high-bit expert model
class MultiBitExpert(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bits * 2, bits * 16),
            nn.ReLU(),
            nn.Linear(bits * 16, bits * 8),
            nn.ReLU(),
            nn.Linear(bits * 8, bits * 4),
            nn.ReLU(),
            nn.Linear(bits * 4, bits),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 🔩 Truth functions
def bitwise_and(a, b): return a & b
def bitwise_or(a, b): return a | b
def bitwise_xor(a, b): return a ^ b
def bitwise_nand(a, b): return (~(a & b)) & 1
def bitwise_not(a, _): return (~a) & 1

# Verifier
def verify(model, truth_fn, bits, is_unary=False):
    model.eval()
    with torch.no_grad():
        for _ in range(25):
            A = torch.randint(0, 2, (bits,), dtype=torch.bool)
            B = torch.randint(0, 2, (bits,), dtype=torch.bool) if not is_unary else torch.zeros_like(A)
            input_tensor = torch.cat([A.float(), B.float()]).unsqueeze(0).to(device)
            expected = truth_fn(A, B).float().to(device)
            output = model(input_tensor).squeeze()
            pred = (output > 0.5).float()
            if not torch.equal(pred, expected):
                return False
    return True

# Training
def train_expert(op_name, bits, truth_fn, is_unary=False):
    fname = f"{op_name}_{bits}bit.pt"
    path = os.path.join(SAVE_DIR, fname)

    if os.path.exists(path):
        model = MultiBitExpert(bits).to(device)
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
            if verify(model, truth_fn, bits, is_unary):
                print(f"SKIPPED: {fname} already verified.")
                return
            else:
                print(f"RETRAINING: {fname} failed verification.")
        except Exception as e:
            print(f"ERROR loading {fname}: {e}. Retraining...")

    def get_training_data():
        x_data, y_data = [], []
        for _ in range(50000):
            A = torch.randint(0, 2, (bits,), dtype=torch.bool)
            B = torch.randint(0, 2, (bits,), dtype=torch.bool) if not is_unary else torch.zeros_like(A)
            input_tensor = torch.cat([A.float(), B.float()])
            target = truth_fn(A, B).float()
            x_data.append(input_tensor)
            y_data.append(target)
        return torch.stack(x_data), torch.stack(y_data)

    print(f"\nTraining {op_name.upper()} expert ({bits} bits)...")
    torch.manual_seed(42)
    x, y = get_training_data()
    x, y = x.to(device), y.to(device)
    train_ds = TensorDataset(x, y)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = MultiBitExpert(bits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCELoss()

    for epoch in range(10000):
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        if epoch % 100 == 0 or avg_loss < 0.01:
            print(f"Epoch {epoch:4d} — Loss: {avg_loss:.6f}")

        # Reseed/reset logic
        if epoch > 5000 and avg_loss > 0.1:
            print(f"LOSS STALLED at epoch {epoch}. Re-seeding and resetting model...")
            torch.manual_seed(random.randint(0, 999999))
            x, y = get_training_data()
            x, y = x.to(device), y.to(device)
            train_ds = TensorDataset(x, y)
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
            model = MultiBitExpert(bits).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.005)

        if avg_loss < 0.0001:
            break

    if verify(model, truth_fn, bits, is_unary):
        torch.save(model, path)
        print(f"SAVED: {fname} | Final Loss: {avg_loss:.6f}")
    else:
        print(f"FAILED: {fname} did not pass verification.")

# Train just 64+ bits
if __name__ == "__main__":
    BIT_WIDTHS = [64, 128, 256, 512]
    OPS = [
        ("and", bitwise_and, False),
        ("or", bitwise_or, False),
        ("xor", bitwise_xor, False),
        ("nand", bitwise_nand, False),
        ("not", bitwise_not, True)
    ]

    for bits in BIT_WIDTHS:
        for name, fn, unary in OPS:
            train_expert(name, bits, fn, unary)