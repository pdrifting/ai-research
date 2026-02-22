import torch
import torch.nn as nn
import torch.optim as optim
import os

SAVE_DIR = "experts"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# Truth Functions (Full Set)
def bitwise_and(a, b): return a & b
def bitwise_or(a, b): return a | b
def bitwise_xor(a, b): return a ^ b
def bitwise_nand(a, b): return (~(a & b)) & 1
def bitwise_nor(a, b): return (~(a | b)) & 1
def bitwise_xnor(a, b): return (a == b)
def bitwise_not(a, _): return (~a) & 1

def verify(model, truth_fn, bits, is_unary=False):
    model.eval()
    with torch.no_grad():
        for _ in range(25):
            A = torch.randint(0, 2, (bits,), dtype=torch.bool)
            B = torch.zeros_like(A) if is_unary else torch.randint(0, 2, (bits,), dtype=torch.bool)
            expected = truth_fn(A, B).float()
            inp = torch.cat([A.float(), B.float()]).unsqueeze(0)
            out = model(inp).squeeze()
            pred = (out > 0.5).float()
            if not torch.equal(pred, expected):
                return False
    return True

def train_expert(op_name, bits, truth_fn, is_unary=False):
    fname = f"{op_name}_{bits}bit.pt"
    path = os.path.join(SAVE_DIR, fname)

    if os.path.exists(path):
        model = MultiBitExpert(bits)
        try:
            check_model = torch.load(path)
            if verify(check_model, truth_fn, bits, is_unary):
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

    sample_count = 10000
    x_data, y_data = [], []
    for _ in range(sample_count):
        A = torch.randint(0, 2, (bits,), dtype=torch.bool)
        B = torch.zeros_like(A) if is_unary else torch.randint(0, 2, (bits,), dtype=torch.bool)
        inp = torch.cat([A.float(), B.float()])
        tgt = truth_fn(A, B).float()
        x_data.append(inp)
        y_data.append(tgt)

    x = torch.stack(x_data)
    y = torch.stack(y_data)

    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        #if epoch % 100 == 0 or loss.item() < 0.01:
        #    print(f"Epoch {epoch:4d} — Loss: {loss.item():.6f}")
        if loss.item() < 0.0001:
            break

    if verify(model, truth_fn, bits, is_unary):
        torch.save(model, path)
        print(f"SAVED: {fname} | Final Loss: {loss.item():.6f}")
    else:
        print(f"REJECTED: {fname} did not pass verification. No file saved.")

# Launch Full Expert Set
if __name__ == "__main__":
    BIT_WIDTHS = range(2, 33)
    OPS = [
        ("and", bitwise_and, False),
        ("or", bitwise_or, False),
        ("xor", bitwise_xor, False),
        ("nand", bitwise_nand, False),
        ("not", bitwise_not, True)
        #("nor", bitwise_nor, False),
        #("xnor", bitwise_xnor, False),
    ]

    for bits in BIT_WIDTHS:
        for name, fn, unary in OPS:
            train_expert(name, bits, fn, unary)
