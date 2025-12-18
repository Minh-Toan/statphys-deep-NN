import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm
import os

matplotlib.rcParams['text.usetex'] = False

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Experiment Parameters 
# ---------------------
d_vals = [200]            # fixed d = 200
alpha = 4.0              # n = alpha * d^2
gamma = 0.5               # teacher gamma
k_factor = gamma
Delta = 0.03            # noise variance
learning_rate = 0.01    
num_epochs = 40000
max_grad_steps = 8000
num_teachers = 10

readout_type = "ones"
activation_type = "relu"   

# student mismatch factors (relative to gamma)
gamma_student_vals = [1.0 * gamma, 2.0 * gamma, 5.0 * gamma, 10.0 * gamma, 20.0 * gamma]

# ---------------------
# Activation helper
# ---------------------
def activate(x):
    if activation_type == "relu":
        return torch.relu(x)
    elif activation_type == "tanh":
        return torch.tanh(2*x)
    else:
        raise ValueError("Invalid activation_type. Use 'relu' or 'tanh'.")

# ---------------------
# Readout function
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
    elif readout_type == 'quaternary':
        choices = torch.tensor([-3., -1., 1., 3.], device=device) / math.sqrt(5)
        probs   = torch.full((4,), 1/4, device=device)
        sample1d = choices[torch.multinomial(probs, k, True)]
        return sample1d.unsqueeze(1)
    else:
        raise ValueError("Invalid readout_type.")

# ---------------------
# Student model
# ---------------------
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

out_dir = "overparam_relu_delta=0.03_alpha=4.0_csv"
os.makedirs(out_dir, exist_ok=True)

for d in d_vals:
    print('dimension =', d)
    k_teacher = int(gamma * d)          # teacher width
    n = int(alpha * d * d)
    print("n/d^2 =", n/d**2)

    # loop over student mismatch settings
    for gamma_student, color in zip(gamma_student_vals, colors):
        k_student = int(gamma_student * d)   # mismatched student width
        print(f"  gamma_student = {gamma_student:.3f}, k_student = {k_student}")

        all_test_losses = []

        for t_run in range(num_teachers):
            # teacher
            Wt = torch.randn(d, k_teacher, device=device) / math.sqrt(d)
            readout = get_readout(k_teacher, readout_type)

            # train data with noise
            Xtr = torch.randn(n, d, device=device)
            hidden_tr = activate(Xtr @ Wt)
            Ytr = hidden_tr @ readout / math.sqrt(k_teacher)
            Ytr += torch.randn(n, 1, device=device) * (Delta**0.5)

            # test data noiseless
            Xte = torch.randn(10000, d, device=device)
            hidden_te = activate(Xte @ Wt)
            Yte = hidden_te @ readout / math.sqrt(k_teacher)

            # student (mismatched width)
            student = Student(d, k_student).to(device)
            opt = optim.Adam(student.parameters(), lr=learning_rate)
            crit = nn.MSELoss()

            test_losses = []
            grad_steps = 0

            for epoch in range(num_epochs):
                student.train()
                perm = torch.randperm(n, device=device)
                chunks = torch.chunk(perm, 5)
                for idx in chunks:
                    if idx.numel() == 0:
                        continue
                    xb, yb = Xtr[idx], Ytr[idx]
                    opt.zero_grad()
                    out = student(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
                    grad_steps += 1

                    student.eval()
                    with torch.no_grad():
                        err = crit(student(Xte), Yte).item()
                    test_losses.append(err)

                    if grad_steps > max_grad_steps:
                        break
                if grad_steps > max_grad_steps:
                    break

            all_test_losses.append(np.array(test_losses))

        L = max(len(tl) for tl in all_test_losses)
        padded = [np.pad(tl, (0, L-len(tl)), constant_values=tl[-1]) for tl in all_test_losses]
        mean_loss = np.mean(padded, axis=0)
        std_loss  = np.std(padded, axis=0)

        xs = np.arange(len(mean_loss))


        out_arr = np.vstack([xs, mean_loss, std_loss]).T
        fname = os.path.join(out_dir, f"results_k_{k_student}.csv")
        np.savetxt(fname, out_arr, delimiter=',',
                   header='grad_update,mean_test_loss,std_test_loss', comments='',
                   fmt='%d,%.8e,%.8e')
        print(f"    saved results to {fname}")

