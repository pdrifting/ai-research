import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime
import secrets
import scipy.stats as stats
import csv
from math import log2

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Custom Dataset for generating true random floats
class RandomFloatDataset(Dataset):
    def __init__(self, size, seq_length=1024):
        self.size = size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        random_bytes = secrets.token_bytes(self.seq_length)
        data = np.frombuffer(random_bytes, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32)

# Generator Network (increased capacity)
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1024):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), # Wider layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.9),
            nn.Linear(1024, 2048), # Wider layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.9),
            nn.Linear(2048, output_dim),
            nn.Tanh() # Output in [-1, 1], scaled to [0, 1] in forward
        )

    def forward(self, x):
        return (self.model(x) + 1) / 2 # Scale [-1, 1] to [0, 1]

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim=1024):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Compute run-length variance for a sequence
def compute_run_length_variance(data):
    binary = (data > 0.5).astype(np.int32)
    runs = []
    current_run = 1
    max_run = 1
    for i in range(1, len(binary)):
        if binary[i] == binary[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    max_run = max(max_run, current_run)
    return np.var(runs) if runs else 0.0, max_run

# Compute approximate entropy (simplified, block length m=2)
def compute_approx_entropy(data, m=2):
    binary = (data > 0.5).astype(np.int32)
    def _phi(m):
        counts = {}
        for i in range(len(binary) - m + 1):
            block = tuple(binary[i:i+m])
            counts[block] = counts.get(block, 0) + 1
        return sum((count / (len(binary) - m + 1)) * log2(count / (len(binary) - m + 1)) for count in counts.values())
    return _phi(m) - _phi(m + 1) if len(binary) >= m + 1 else 0.0

# Function to compute metrics
def compute_metrics(generator, batch_size=1024):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(batch_size, 100).to(device)
        generated = generator(noise).cpu().numpy().flatten()
        while len(generated) < 1048576:
            noise = torch.randn(batch_size, 100).to(device)
            more_data = generator(noise).cpu().numpy().flatten()
            generated = np.concatenate([generated, more_data])
        generated = generated[:1048576]
        variance = stats.variation(generated, axis=0)
        run_var, max_run = compute_run_length_variance(generated)
        approx_entropy = compute_approx_entropy(generated)
    generator.train()
    return variance, run_var, max_run, approx_entropy

# Function to save 1MB byte stream
def save_byte_stream(generator, cycle, epoch, output_dir, run_var):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1024, 100).to(device)
        generated = generator(noise).cpu().numpy()
        byte_stream = (generated * 255).astype(np.uint8).flatten()
        while len(byte_stream) < 1048576:
            noise = torch.randn(1024, 100).to(device)
            more_bytes = (generator(noise).cpu().numpy() * 255).astype(np.uint8).flatten()
            byte_stream = np.concatenate([byte_stream, more_bytes])
        byte_stream = byte_stream[:1048576]
        filename = os.path.join(output_dir, f"byte_stream_cycle_{cycle}_epoch_{epoch}_runvar_{run_var:.6f}.bin")
        with open(filename, "wb") as f:
            f.write(byte_stream.tobytes())
    generator.train()
    return filename

# Training function
def train_cycle(cycle, num_epochs, generator, discriminator, dataloader, g_optimizer, d_optimizer, output_dir, log_writer, prev_checkpoint=None, target_loss=1.75):
    criterion = nn.BCELoss()
    best_run_var = -float("inf")
    best_epoch = 0

    if prev_checkpoint:
        checkpoint = torch.load(prev_checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print(f"Loaded weights from cycle {checkpoint['cycle']}, epoch {checkpoint['epoch']}")

    for epoch in range(num_epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        batches = 0

        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            noise = torch.randn_like(real_data) * 0.01
            real_data_noisy = (real_data + noise).clamp(0, 1)

            # Label flipping (5% chance)
            if np.random.rand() < 0.05:
                real_labels = torch.zeros(batch_size, 1).to(device)
                fake_labels = torch.ones(batch_size, 1).to(device) * 0.9
            else:
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            d_real = discriminator(real_data_noisy)
            d_real_loss = criterion(d_real, real_labels)
            real_run_var, _ = compute_run_length_variance(real_data.cpu().numpy().flatten())
            real_approx_entropy = compute_approx_entropy(real_data.cpu().numpy().flatten())

            noise = torch.randn(batch_size, 100).to(device)
            fake_data = generator(noise)
            d_fake = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake, fake_labels)
            fake_run_var, _ = compute_run_length_variance(fake_data.detach().cpu().numpy().flatten())
            fake_approx_entropy = compute_approx_entropy(fake_data.detach().cpu().numpy().flatten())

            # Gradient penalty
            alpha = torch.rand(batch_size, 1).to(device)
            alpha = alpha.expand_as(real_data)
            interpolates = (alpha * real_data + (1 - alpha) * fake_data.detach()).requires_grad_(True)
            d_interpolates = discriminator(interpolates)
            gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones_like(d_interpolates),
                                            create_graph=True, retain_graph=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

            # Discriminator loss with RLV and entropy penalties
            run_var_diff = torch.tensor(abs(real_run_var - fake_run_var), dtype=torch.float32).to(device)
            entropy_diff = torch.tensor(abs(real_approx_entropy - fake_approx_entropy), dtype=torch.float32).to(device)
            d_loss = d_real_loss + d_fake_loss + 0.3 * run_var_diff + 0.1 * entropy_diff + gradient_penalty

            d_loss.backward()
            d_optimizer.step()

            # Train Generator (3 updates)
            for _ in range(2):
                g_optimizer.zero_grad()
                fake_data = generator(noise)
                d_fake = discriminator(fake_data)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                g_optimizer.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            batches += 1

        avg_d_loss = d_loss_total / batches
        avg_g_loss = g_loss_total / batches
        variance, run_var, max_run, approx_entropy = compute_metrics(generator)

        # Log metrics
        log_writer.writerow([cycle, epoch + 1, avg_d_loss, avg_g_loss, variance, run_var, max_run, approx_entropy])
        print(f"Cycle {cycle}, Epoch {epoch+1}/{num_epochs}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, Variance: {variance:.6f}, RLV: {run_var:.6f}, Max Run: {max_run}, Entropy: {approx_entropy:.6f}")

        # Save model if RLV is best and variance is within acceptable range (0.52–0.63)
        if run_var > best_run_var and 0.52 <= variance <= 0.63:
            best_run_var = run_var
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(output_dir, f"checkpoint_cycle_{cycle}_epoch_{best_epoch}_runvar_{best_run_var:.6f}.pt")
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'cycle': cycle,
                'epoch': best_epoch,
                'g_loss': avg_g_loss,
                'variance': variance,
                'run_var': best_run_var,
                'max_run': max_run,
                'approx_entropy': approx_entropy
            }, checkpoint_path)
            byte_stream_file = save_byte_stream(generator, cycle, best_epoch, output_dir, best_run_var)
            print(f"Saved checkpoint and byte stream: {byte_stream_file}")

        if abs(avg_g_loss - target_loss) < 0.05:
            print(f"Target loss {target_loss} reached at Cycle {cycle}, Epoch {epoch+1}")
            break

    return checkpoint_path

def main():
    batch_size = 64
    input_dim = 100
    output_dim = 1024
    num_cycles = 3
    epochs_per_cycle = [50, 50, 50]
    lr = 0.00005 # Reduced for stability
    momentum = 0.9

    output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cycle', 'epoch', 'd_loss', 'g_loss', 'variance', 'run_length_variance', 'max_run_length', 'approx_entropy'])

    dataset = RandomFloatDataset(size=10000, seq_length=output_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    generator = Generator(input_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)
    g_optimizer = optim.RMSprop(generator.parameters(), lr=lr, momentum=momentum)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr, momentum=momentum)

    prev_checkpoint = None
    for cycle in range(1, num_cycles + 1):
        print(f"\nStarting Cycle {cycle}")
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            prev_checkpoint = train_cycle(cycle, epochs_per_cycle[cycle-1], generator, discriminator, dataloader, g_optimizer, d_optimizer, output_dir, writer, prev_checkpoint)

if __name__ == "__main__":
    main()
