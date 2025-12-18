import os, time, math, csv, gc
import numpy as np
import torch

# ---------------------
# Device
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# Fixed parameters
# ---------------------
d = 150
gamma = 0.5
k = int(gamma * d)    # teacher hidden width
Delta = 0.1           # noise variance for training
num_test = 10000

# ---------------------
# Experiment control
# ---------------------
alpha_vals = np.arange(0.25, 8.25 + 1e-9, 0.5).tolist()  
num_teacher_runs = 10
lambda_reg = 1e-3       # kernel ridge regularization (lambda)
cg_tol = 1e-6
cg_maxiter = 2000
chol_n_threshold = 4000  # <= this -> Cholesky direct solve

activation_type = "relu"   # "relu" or "tanh"

# ---------------------
# STUDENT random-features width control
# ---------------------
# set student_width_multiplier (beta). student width w = int(beta * k * d)
student_width_multiplier = 3.0
print("student_width_multiplier (beta) =", student_width_multiplier)

# output paths
out_dir = "krr_student_features_results_relu"
os.makedirs(out_dir, exist_ok=True)
master_csv = os.path.join(out_dir, f"krr_student_beta{student_width_multiplier:.3f}_lambda{lambda_reg:.6f}.csv")

# ---------------------
# Helpers
# ---------------------
def activate_torch(x):
    if activation_type == "relu":
        return torch.relu(x)
    elif activation_type == "tanh":
        return torch.tanh(2.0 * x)
    else:
        raise ValueError("Invalid activation_type")

def activate_numpy(x):
    if activation_type == "relu":
        return np.maximum(0.0, x)
    elif activation_type == "tanh":
        return np.tanh(2.0 * x)
    else:
        raise ValueError("Invalid activation_type")

# ---------- memory-block helpers ----------
def _rows_per_block(w, target_elems=1_000_000):
    """Return a number of rows per block so that block size ~ target_elems floats."""
    # target_elems floats (float32) ~= 4 bytes each -> but we pick in floats count
    return max(1, int(max(1, target_elems // max(1, w))))

def rf_phi_T_v_blockwise(X_t, W_student_t, v, activation_fn, w, rows_per_block=None):
    """
    Compute t = Phi^T v in blocks WITHOUT forming full Phi.
    X_t: (n,d), W_student_t: (d,w), v: (n,)
    returns t: (w,)
    """
    if rows_per_block is None:
        rows_per_block = _rows_per_block(w)
    device = X_t.device
    t = torch.zeros(w, device=device, dtype=X_t.dtype)
    n = X_t.shape[0]
    scale = 1.0 / math.sqrt(w)
    for i in range(0, n, rows_per_block):
        xb = X_t[i:i+rows_per_block]               # (b,d)
        vb = v[i:i+rows_per_block]                 # (b,)
        block = activation_fn(xb @ W_student_t) * scale  # (b,w)
        t += block.t().mv(vb)                      # (w,)
        del block, xb, vb
    return t

def rf_phi_mv_blockwise(X_t, W_student_t, u, activation_fn, w, rows_per_block=None):
    """
    Compute out = Phi u in blocks WITHOUT forming full Phi.
    returns out: (n,)
    """
    if rows_per_block is None:
        rows_per_block = _rows_per_block(w)
    device = X_t.device
    n = X_t.shape[0]
    out = torch.empty(n, device=device, dtype=X_t.dtype)
    scale = 1.0 / math.sqrt(w)
    for i in range(0, n, rows_per_block):
        xb = X_t[i:i+rows_per_block]               # (b,d)
        block = activation_fn(xb @ W_student_t) * scale  # (b,w)
        out[i:i+rows_per_block] = block.mv(u)           # (b,)
        del block, xb
    return out

# ---------------------
# Kernel ridge solver 
# ---------------------
def kernel_ridge_solve(Phi=None, y=None, lambda_reg=lambda_reg, tol=cg_tol, maxiter=cg_maxiter,
                       chol_thr=chol_n_threshold, x0=None,
                       X_t=None, W_student_t=None, activation_fn=None, rows_per_block=None):
    """
    Two modes:
     - classical: pass Phi (torch tensor n x w) and y -> old behavior
     - memory-efficient: pass X_t (n,d), W_student_t (d,w) and activation_fn; Phi must be None.
    Returns alpha (n,) on device.
    """
    if Phi is None and (X_t is None or W_student_t is None or activation_fn is None):
        raise ValueError("Either Phi or (X_t, W_student_t, activation_fn) must be provided")

    if Phi is not None:
        n = Phi.shape[0]
        # Use Cholesky when small
        if n <= chol_thr:
            K = Phi @ Phi.t()                             # (n,n)
            A = K + lambda_reg * torch.eye(n, device=Phi.device, dtype=Phi.dtype)
            L = torch.linalg.cholesky(A)
            alpha = torch.cholesky_solve(y.unsqueeze(1), L).reshape(n)
            return alpha

        # matrix-free matvec using Phi
        def A_matvec(v):
            u = Phi.t().mv(v)          # (w,)
            Kv = Phi.mv(u)            # (n,)
            return Kv + lambda_reg * v

        if x0 is None:
            x = torch.zeros_like(y, device=Phi.device)
        else:
            x = x0.clone().to(Phi.device)

        r = y - A_matvec(x)
        p = r.clone()
        rsold = (r * r).sum()
        if rsold.sqrt() <= tol:
            return x
        for i in range(maxiter):
            Ap = A_matvec(p)
            pAp = (p * Ap).sum()
            if pAp.abs() < 1e-30:
                break
            alpha_cg = rsold / pAp
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap
            rsnew = (r * r).sum()
            if rsnew.sqrt().item() <= tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    # ---------- memory-efficient path using X_t, W_student_t ----------
    n = X_t.shape[0]
    device = X_t.device
    dtype = X_t.dtype
    w = W_student_t.shape[1]

    # If n small enough, it's cheaper to materialize Phi and use Cholesky
    if n <= chol_thr:
        scale = 1.0 / math.sqrt(w)
        Phi_full = activation_fn(X_t @ W_student_t) * scale   # (n,w)
        K = Phi_full @ Phi_full.t()
        A = K + lambda_reg * torch.eye(n, device=device, dtype=dtype)
        L = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).reshape(n)
        del Phi_full, K, A, L
        torch.cuda.empty_cache()
        return alpha

    # matrix-free CG using blockwise matvecs
    rpblock = rows_per_block if rows_per_block is not None else _rows_per_block(w)
    def A_matvec(v):
        t = rf_phi_T_v_blockwise(X_t, W_student_t, v, activation_fn, w, rows_per_block=rpblock)  # (w,)
        Kv = rf_phi_mv_blockwise(X_t, W_student_t, t, activation_fn, w, rows_per_block=rpblock)  # (n,)
        return Kv + lambda_reg * v

    if x0 is None:
        x = torch.zeros_like(y, device=device)
    else:
        x = x0.clone().to(device)

    r = y - A_matvec(x)
    p = r.clone()
    rsold = (r * r).sum()
    if rsold.sqrt() <= tol:
        return x
    for i in range(maxiter):
        Ap = A_matvec(p)
        pAp = (p * Ap).sum()
        if pAp.abs() < 1e-30:
            break
        alpha_cg = rsold / pAp
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        rsnew = (r * r).sum()
        if rsnew.sqrt().item() <= tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# ---------------------
# Data generation (teacher) -- teacher W and r are unknown to the student
# ---------------------
def generate_teacher_data(n, seed=None):
    """Return teacher W (d,k), r (k,1), X (n,d), Y (n,1, noisy), and test noiseless Xte,Yte (numpy)"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    # Teacher parameters (unknown to student)
    W_teacher = (np.random.randn(d, k).astype(np.float32) / math.sqrt(d))   # (d,k)
    r_teacher = (np.random.randn(k, 1).astype(np.float32))                 # Gaussian readout

    X = np.random.randn(n, d).astype(np.float32)                           # train inputs
    H = activate_numpy(X @ W_teacher)   # teacher hidden (n,k)
    Y = (H @ r_teacher / math.sqrt(k)).astype(np.float32)                  # noiseless teacher outputs
    # add training noise
    Y += (np.random.randn(*Y.shape).astype(np.float32) * math.sqrt(Delta))

    # test noiseless labels 
    Xte = np.random.randn(num_test, d).astype(np.float32)
    Hte = activate_numpy(Xte @ W_teacher)
    Yte = (Hte @ r_teacher / math.sqrt(k)).astype(np.float32)
    return W_teacher, r_teacher, X, Y, Xte, Yte

# ---------------------
# Prepare master CSV 
# ---------------------
if not os.path.exists(master_csv):
    with open(master_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha","run","test_mse","solve_time_s","n","student_w","chosen_lambda"])

# track done runs to skip on restart
done = {}  # done[alpha_str] = max run index recorded
if os.path.exists(master_csv):
    with open(master_csv, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            a_str = str(row[0]).strip()
            try:
                run_idx = int(float(row[1]))
            except Exception:
                continue
            done[a_str] = max(done.get(a_str, 0), run_idx)

# ---------------------
# Parameters for lambda search
# ---------------------
lambda_candidates = np.array([1e-2,2.5e-2,5e-2,7.5e-2,1e-1,2.5e-1,5e-1,7.5e-1,1.0,1.25,1.5], dtype=np.float32)
val_frac = 0.10
train_sub_max = 2000   # max size of training subproblem used in search
tol_search = 1e-4
maxiter_search = 200

MEM_EFF_THRESHOLD = 50_000_000  
BLOCK_TARGET_ELEMS = 1_000_000  

# ---------------------
# Main sweep
# ---------------------
for alpha_val in alpha_vals:
    alpha_str = f"{alpha_val:.6f}"
    n_done = done.get(alpha_str, 0)
    n = int(alpha_val * d * d)
    print(f"\n=== alpha={alpha_val:.6f}, n={n}, skipping first {n_done} runs ===")

    per_run_mses = []
    for run in range(n_done + 1, num_teacher_runs + 1):
        seed = 1000 + run + int(round(alpha_val*1000))
        print(f" run {run}/{num_teacher_runs} (seed={seed}) ...", end=" ", flush=True)

        # -----------------
        # Teacher generates data 
        # -----------------
        W_teacher, r_teacher, X, Y, Xte, Yte = generate_teacher_data(n, seed=seed)

        X_t = torch.from_numpy(X).to(device)
        Xte_t = torch.from_numpy(Xte).to(device)
        y = torch.from_numpy(Y.reshape(n)).to(device)
        y_test = torch.from_numpy(Yte.reshape(-1)).to(device)

        # -----------------
        # Build student random features
        # -----------------
        w = int(max(1, round(student_width_multiplier * k * d)))  # student feature width
        if w > 5e7:
            raise MemoryError(f"student width w={w} is extremely large; reduce student_width_multiplier")

        W_student_t = (torch.randn(d, w, device=device, dtype=torch.float32) / math.sqrt(d))

        use_mem_efficient = (n * w) > MEM_EFF_THRESHOLD

        if not use_mem_efficient:
            scale = 1.0 / math.sqrt(w)
            Phi = activate_torch(X_t @ W_student_t) * scale   # (n,w)
        else:
            Phi = None

        # -----------------
        # --- lambda search ---
        # -----------------
        chosen_lambda = lambda_reg
        if n > 20:
            val_n = max(min(int(math.ceil(val_frac * n)), 1000), 10)
            train_sub_n = min(max(200, n - val_n), train_sub_max)
            idx_all = np.arange(n)
            rng = np.random.default_rng(seed + 12345)
            val_idx = rng.choice(idx_all, size=val_n, replace=False)
            remaining = np.setdiff1d(idx_all, val_idx)
            if remaining.size == 0:
                train_sub_idx = val_idx
            else:
                train_sub_idx = rng.choice(remaining, size=min(train_sub_n, remaining.size), replace=False)

            train_sub_idx_t = torch.from_numpy(train_sub_idx.astype(np.int64)).to(device)
            val_idx_t = torch.from_numpy(val_idx.astype(np.int64)).to(device)

            if use_mem_efficient:
                Phi_train_sub = activate_torch(X_t[train_sub_idx_t] @ W_student_t) / math.sqrt(w)
                Phi_val = activate_torch(X_t[val_idx_t] @ W_student_t) / math.sqrt(w)
                y_train_sub = y.index_select(0, train_sub_idx_t)
                y_val = y.index_select(0, val_idx_t)
            else:
                Phi_train_sub = Phi.index_select(0, train_sub_idx_t)
                Phi_val = Phi.index_select(0, val_idx_t)
                y_train_sub = y.index_select(0, train_sub_idx_t)
                y_val = y.index_select(0, val_idx_t)

            best_val_mse = float('inf')
            best_lambda = lambda_reg
            for lam in lambda_candidates:
                lam = float(lam)
                # solve subproblem using existing kernel_ridge_solve 
                alpha_sub = kernel_ridge_solve(Phi=Phi_train_sub, y=y_train_sub, lambda_reg=lam,
                                               tol=tol_search, maxiter=maxiter_search, chol_thr=chol_n_threshold)
                # produce predictions on validation set
                u_sub = Phi_train_sub.t().mv(alpha_sub)   # (w,)
                y_val_pred = Phi_val.mv(u_sub)
                val_mse = float(((y_val_pred - y_val).pow(2).mean()).item())
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_lambda = lam
            chosen_lambda = best_lambda
            print(f"(lambda search -> chosen {chosen_lambda:.1e}, val_mse={best_val_mse:.3e})", end=" ", flush=True)
            if use_mem_efficient:
                del Phi_train_sub, Phi_val, y_train_sub, y_val
                torch.cuda.empty_cache()

        # -----------------
        # Solve KRR with student's kernel K_student = Phi Phi^T using chosen_lambda
        # -----------------
        t0 = time.time()
        if use_mem_efficient:
            alpha_dual = kernel_ridge_solve(Phi=None, y=y, lambda_reg=chosen_lambda, tol=cg_tol, maxiter=cg_maxiter,
                                            chol_thr=chol_n_threshold, x0=None,
                                            X_t=X_t, W_student_t=W_student_t,
                                            activation_fn=lambda z: activate_torch(z),
                                            rows_per_block=_rows_per_block(w, BLOCK_TARGET_ELEMS))
        else:
            alpha_dual = kernel_ridge_solve(Phi=Phi, y=y, lambda_reg=chosen_lambda, tol=cg_tol, maxiter=cg_maxiter, chol_thr=chol_n_threshold)
        solve_time = time.time() - t0

        # -----------------
        # Predict using student's features (compute u = Phi^T alpha then test predictions)
        # -----------------
        if use_mem_efficient:
            u = rf_phi_T_v_blockwise(X_t, W_student_t, alpha_dual, activation_fn=lambda z: activate_torch(z), w=w,
                                     rows_per_block=_rows_per_block(w, BLOCK_TARGET_ELEMS))
            # test predictions blockwise
            rows_per_block = _rows_per_block(w, BLOCK_TARGET_ELEMS)
            ntest = Xte_t.shape[0]
            y_pred = torch.empty(ntest, device=Xte_t.device, dtype=Xte_t.dtype)
            scale = 1.0 / math.sqrt(w)
            for i in range(0, ntest, rows_per_block):
                xb = Xte_t[i:i+rows_per_block]
                block = activate_torch(xb @ W_student_t) * scale  # (b,w)
                y_pred[i:i+rows_per_block] = block.mv(u)
                del block, xb
        else:
            u = Phi.t().mv(alpha_dual)   # (w,)
            Phi_test = activate_torch(Xte_t @ W_student_t) / math.sqrt(w)
            y_pred = Phi_test.mv(u)
            del Phi_test

        test_mse = float(((y_pred - y_test).pow(2).mean()).item())

        run_fname = os.path.join(out_dir, f"krr_student_alpha{alpha_str}_run{run}.npz")
        np.savez_compressed(run_fname,
                            test_mse=float(test_mse),
                            solve_time_s=float(solve_time),
                            seed=int(seed),
                            alpha=float(alpha_val),
                            n=int(n),
                            student_w=int(w),
                            chosen_lambda=float(chosen_lambda))
        with open(master_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([alpha_str, run, f"{test_mse:.6e}", f"{solve_time:.4f}", n, w, f"{chosen_lambda:.3e}"])
        print(f"done (mse={test_mse:.4e}, time={solve_time:.2f}s, student_w={w})")

        per_run_mses.append(test_mse)
        done[alpha_str] = max(done.get(alpha_str, 0), run)
        # free large tensors for this run
        try:
            del X, Xte, Y, Yte
        except Exception:
            pass
        try:
            del X_t, Xte_t, y, y_test, W_student_t, alpha_dual, u, y_pred
        except Exception:
            pass
        if 'Phi' in locals():
            try:
                del Phi
            except Exception:
                pass
        torch.cuda.empty_cache()
        gc.collect()

    if len(per_run_mses) > 0:
        mean_mse = np.mean(per_run_mses)
        std_mse = np.std(per_run_mses, ddof=0)
        print(f" alpha {alpha_val:.6f}: mean_mse={mean_mse:.6e}, std_mse={std_mse:.6e} over {len(per_run_mses)} runs")
    else:
        print(f" alpha {alpha_val:.6f}: no new runs (all previously done)")

print("\nSweep finished. Master CSV:", master_csv)