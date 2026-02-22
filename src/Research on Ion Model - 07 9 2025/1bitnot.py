import torch
import torch.nn as nn
import torch.optim as optim
import os

SAVE_DIR = "experts"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "not_1bit.pt"
SAVE_PATH = os.path.join(SAVE_DIR, MODEL_NAME)

# Expert model for 1-bit NOT gate (1 input, 1 output)
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

# NOT logic: 0 → 1, 1 → 0
def bitwise_not(a): return (~a.bool()) & 1

# Verify model matches truth table
def verify(model):
    model.eval()
    with torch.no_grad():
        for a in [0, 1]:
            truth = bitwise_not(torch.tensor([a])).float()
            inp = torch.tensor([[float(a)]], dtype=torch.float32)
            out = model(inp).item()
            pred = 1.0 if out > 0.5 else 0.0
            if pred != truth.item():
                print(f"❌ FAIL: NOT({a}) → Model: {int(pred)} vs Truth: {int(truth.item())}")
                return False
    return True

# Train NOT expert
def train_not_expert():
    if os.path.exists(SAVE_PATH):
        print(f"Checking existing model: {MODEL_NAME}")
        check = torch.load(SAVE_PATH)
        if verify(check):
            print(f"VERIFIED: {MODEL_NAME} already trained and correct.")
            return
        else:
            print(f"Existing model failed verification. Retraining...")

    print(f"\nTraining NOT expert (1-bit)...")
    torch.manual_seed(42)
    model = NotExpert1Bit()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    # 1 input (1-bit), 1 output
    x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y_train = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0001:
            break

    if verify(model):
        torch.save(model, SAVE_PATH)
        print(f"SAVED: {MODEL_NAME} | Final Loss: {loss.item():.6f}")
    else:
        print(f"Training complete but model failed verification. Not saved.")

# Entry point
if __name__ == "__main__":
    train_not_expert()
