import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv  

# Use GPU if available
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# Experiment Parameters
# ---------------------
d_vals = [80,100,120,140,160,220]  # Dimensions to test
alpha = 5                    # so that n = alpha * d^2
gamma = 0.5                  # so that k = gamma * d
Delta = 0.0001               # noise variance
learning_rate = 0.01         # fixed learning rate
num_epochs = 100000            # maximum epochs per teacher run
max_grad_steps = 25000        # maximum gradient steps allowed
num_teachers = 10             # number of teacher runs per dimension
num_test = 10000                # number of test samples

readout_type = "ones"
csv_filename = "LEARNABLE_TANH_mean_test_losses_ones_lr=0.01.csv"

def activate(x):
    return torch.tanh(2*x)

# ---------------------
# Derived Parameters and Model Definition
# ---------------------
def get_readout(k, readout_type):
    if readout_type == "gaussian":
        return torch.randn(k, 1, device=device)
    elif readout_type == "rademacher":
        return torch.randint(0, 2, (k, 1), device=device, dtype=torch.float32) * 2 - 1
    elif readout_type == "uniform":
        a = math.sqrt(3)
        return torch.rand(k, 1, device=device) * (2 * a) - a
    elif readout_type == "ones":
        return torch.ones(k, 1, device=device)
    else:
        raise ValueError("Invalid readout_type. Choose from 'gaussian', 'rademacher', 'uniform', or 'ones'.")

class Student(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc1 = nn.Linear(d, k, bias=False)
        with torch.no_grad():
            self.fc1.weight.div_(math.sqrt(d))
        # Initialize the student's readout randomly and make it learnable
        self.readout = nn.Parameter(get_readout(k, readout_type))
    def forward(self, x):
        h = activate(self.fc1(x))
        return h @ self.readout / math.sqrt(h.size(1))

# ---------------------
# Data generation
# ---------------------
def generate_teacher_data(n):
    W = torch.randn(d, k, device=device)/math.sqrt(d)
    X = torch.randn(n, d, device=device)
    h = activate(X @ W)
    readout = get_readout(k, readout_type)
    Y = h @ readout / math.sqrt(k)
    Y += torch.randn(n, 1, device=device)*math.sqrt(Delta)
    return W, readout, X, Y

def generate_test_data(W, readout):
    X = torch.randn(num_test, d, device=device)
    h = activate(X @ W)
    Y = h @ readout / math.sqrt(k)
    return X, Y

# ---------------------
# Check existing CSV and determine missing d's
# ---------------------
existing_ds = set()
if os.path.exists(csv_filename):
    try:
        with open(csv_filename, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  
            for row in reader:
                if not row:
                    continue
                try:
                    existing_ds.add(int(float(row[0])))
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: couldn't read existing CSV '{csv_filename}': {e}")

process_d_vals = [dd for dd in d_vals if dd not in existing_ds]
print(f"Existing d in CSV: {sorted(existing_ds) if existing_ds else 'None'}")
print(f"Will process (missing) d values: {process_d_vals}")

if len(process_d_vals) == 0:
    print("Nothing to compute — all specified d values are already present in the CSV.")
    raise SystemExit("All dimensions already saved. Exiting.")

# ---------------------
# Average Over Teacher instances
# ---------------------
plt.figure(figsize=(10, 6))
small = 12  

colors = [cm.Reds(i) for i in np.linspace(0.3, 1, len(process_d_vals))]  # Reds colormap

for d, color in zip(process_d_vals, colors):
    k = int(gamma * d)
    n = int(alpha * d**2)
    batch_size = max(1, int(n // 4))  # Fixed batch size = n/4
    
    print(f"\nDimension d = {d}: using batch size = {batch_size}")
    
    all_test_losses = []

    for t_run in range(num_teachers):
        print('     teacher run =',t_run)
        # Generate teacher data with a Gaussian readout
        W_teacher, readout, X_train, Y_train = generate_teacher_data(n)
        X_test, Y_test = generate_test_data(W_teacher, readout)
        
        student = Student(d, k).to(device)
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        test_losses = []
        grad_steps = 0
        
        for epoch in range(num_epochs):
            student.train()
            permutation = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                indices = permutation[i:i+batch_size]
                batch_X = X_train[indices]
                batch_Y = Y_train[indices]
                
                optimizer.zero_grad()
                outputs = student(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                grad_steps += 1
                
                student.eval()
                with torch.no_grad():
                    current_loss = criterion(student(X_test), Y_test).item()
                test_losses.append(current_loss)
                
                if grad_steps > max_grad_steps:
                    break
            if grad_steps > max_grad_steps:
                break
        
        all_test_losses.append(test_losses)

    max_length = max(len(tl) for tl in all_test_losses)
    padded_losses = [np.pad(tl, (0, max_length - len(tl)), constant_values=tl[-1]) for tl in all_test_losses]
    
    mean_loss = np.mean(padded_losses, axis=0)
    std_loss = np.std(padded_losses, axis=0)

    rows_to_write = []
    for grad_step in range(len(mean_loss)):
        rows_to_write.append([d, int(grad_step), float(mean_loss[grad_step]), float(std_loss[grad_step])])

    if os.path.exists(csv_filename):
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)
    else:
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["d", "gradient_step", "mean_loss", "std_loss"])
            writer.writerows(rows_to_write)
    print(f"Saved results for d = {d} to '{csv_filename}'")