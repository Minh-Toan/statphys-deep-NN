import csv
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
from jax import random
import math
import scipy.integrate
import gc
import time
import json

# RNG per-process
BASE_SEED = 1234
rng_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", False)
devices = jax.devices()

# ----------------------
# Hyperparameters
# ----------------------
d = 200                   # input dimension
gamma1 = 0.5              # k1/d
# alphas (example)
#alphas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0]
alphas = [1.0,1.5,3.0,6.0]
delta = 0.1                 # noise variance
sigma = jnp.sqrt(delta)      
info_init = 1                    # whether to use informative initialization
quenched = 0                    # whether to use quenched student readout 
lr = 1                          # whether to learn readout weights
tree_depth = 7                  # max tree depth for NUTS

num_teacher = 12                # number of teacher realizations per alpha
acc_rate = 0.65                 # target acceptance rate for NUTS

n_test = 100000                 # number of test samples for Gibbs error estimation

# ---------- Activation ----------
def activation(x):
    return (jax.nn.relu(x))
# ---------------------------------

# Teacher forward for single-hidden-layer
def make_teacher_forward_1l(W1_teacher, v_teacher, d, k1):
    def teacher_forward(x):
        h1 = activation((1.0/jnp.sqrt(d)) * x @ W1_teacher.T)   # (n, k1)
        return (1.0/jnp.sqrt(k1)) * (h1 @ v_teacher)            # (n,)
    return teacher_forward

###########################
# Overall Results Storage #
###########################
results_csv_filename = f"relu_3_2LNN_info_gauss_gibbs_errors_d={d}_delta{int(delta*100)}.csv"
single_diags_filename = f"sample_3_relu_2LNN_info_diags_gauss_d={d}.npz"

def save_results_to_csv(results, csv_filename, num_teacher):
    tmp = csv_filename + f".tmp{os.getpid()}"
    header = ["alpha"] + [f"teacher_{i+1}" for i in range(num_teacher)]
    with open(tmp, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for alpha in sorted(results.keys()):
            teacher_results = results[alpha]
            row = [alpha] + teacher_results + ["" for _ in range(num_teacher - len(teacher_results))]
            csv_writer.writerow(row)
    os.replace(tmp, csv_filename)
    print(f"[proc {_proc_idx}] Saved current overall results to {csv_filename}")

def load_results_from_csv(csv_filename, alphas, num_teacher):
    results = {a: [] for a in alphas}
    if os.path.exists(csv_filename):
        with open(csv_filename, mode="r", newline="") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    alpha_val = float(row["alpha"])
                except ValueError:
                    continue
                teacher_values = []
                for i in range(num_teacher):
                    key = f"teacher_{i+1}"
                    if row.get(key, "") != "":
                        try:
                            teacher_values.append(float(row[key]))
                        except ValueError:
                            pass
                results[alpha_val] = teacher_values
        print(f"[proc {_proc_idx}] Loaded previous overall results from {csv_filename}")
    else:
        print(f"[proc {_proc_idx}] No previous overall results found at {csv_filename}.")
    return results

results = load_results_from_csv(results_csv_filename, alphas, num_teacher)

# ------------------------------
# Single-file diagnostics utils 
# ------------------------------
def _init_empty_storage_1l(k1, d):
    return dict(
        diag_W1 = np.zeros((0, k1), dtype=np.float32),
        diag_W1_feature = np.zeros((0, d), dtype=np.float32),
        v_teacher = np.zeros((0, k1), dtype=np.float32),
        s_sample = np.zeros((0, k1), dtype=np.float32),   # student readout samples
        R2 = np.zeros((0,), dtype=np.float32),            
        # single-sample
        last_W1_sample = np.zeros((0, k1, d), dtype=np.float32),
        last_v_sample = np.zeros((0, k1), dtype=np.float32),
        W1_teacher_store = np.zeros((0, k1, d), dtype=np.float32),
        v_teacher_store = np.zeros((0, k1), dtype=np.float32),
        alpha = np.zeros((0,), dtype=np.float32),
        teacher_idx = np.zeros((0,), dtype=np.int32),
        gibbs_error = np.zeros((0,), dtype=np.float32),
    )

def save_append_single_file_1l(filename, new_entry, k1, d):
    if os.path.exists(filename):
        with np.load(filename, allow_pickle=False) as data:
            stored = {k: data[k] for k in data.files}
        storage = {}
        defaults = _init_empty_storage_1l(k1, d)
        for key in defaults.keys():
            storage[key] = stored.get(key, defaults[key])
        for k in new_entry:
            storage[k] = np.concatenate((storage[k], new_entry[k]), axis=0)
    else:
        storage = _init_empty_storage_1l(k1, d)
        for k in new_entry:
            storage[k] = new_entry[k]

    try:
        with open(filename, "wb") as f:
            np.savez_compressed(f, **storage)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            pass
        raise RuntimeError(f"[proc {_proc_idx}] Failed writing {filename}: {e}") from e

    print(f"[proc {_proc_idx}] Appended diagnostics into {filename}")

print(f"[proc {_proc_idx}] Started!")
for alpha in alphas:
    # warmup schedule 
    if (alpha < 3.0):
       num_warmup_samples = 6000
    elif (alpha < 5.0):
       num_warmup_samples = 4500
    elif (alpha < 8.0):
       num_warmup_samples = 3000

    if alpha not in results:
        results[alpha] = []
    current_done = len(results[alpha])
    if current_done >= num_teacher:
        print(f"[proc {_proc_idx}] Alpha {alpha} already has {current_done} teacher realizations. Skipping...")
        continue

    n = int(alpha * d**2)

    for t in range(current_done, num_teacher):
        rng_key, key_teacher, key_data, key_mcmc, key_test = jax.random.split(rng_key, 5)
        key_W1, key_v, key_X, key_noise = jax.random.split(key_teacher, 4)

        k1 = int(gamma1 * d)

        W1_teacher = jax.random.normal(key_W1, (k1, d), dtype=jnp.float32)

        # Teacher readout vector
        v_teacher = np.mean(np.sort(np.random.randn(100, k1), axis=1), axis=0).astype(np.float32)

        teacher_forward = make_teacher_forward_1l(W1_teacher, jnp.asarray(v_teacher), d, k1)

        X = jax.random.normal(key_X, (n, d), dtype=jnp.float32)
        y = teacher_forward(X) + sigma * jax.random.normal(key_noise, (n,), dtype=jnp.float32)

        # Student model for single hidden layer
        if lr:
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0,1).expand([k1, d]).to_event(2))
                if quenched:
                    v = numpyro.sample("v", dist.Normal(jnp.array(v_teacher), 0.0001).expand([k1]).to_event(1))
                else:
                    v = numpyro.sample("v", dist.Normal(0,1).expand([k1]).to_event(1))
                h1 = activation((1.0/jnp.sqrt(d)) * X @ W1.T)
                f = (1.0/jnp.sqrt(k1)) * (h1 @ v)
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)
        else:
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0,1).expand([k1, d]).to_event(2))
                h1 = activation((1.0/jnp.sqrt(d)) * X @ W1.T)
                f = (1.0/jnp.sqrt(k1)) * (h1 @ jnp.asarray(v_teacher))
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)

        # init params
        if lr:
            if info_init:
                init_params = {"W1": W1_teacher, "v": jnp.asarray(v_teacher)}
            else:
                init_key1, init_key2 = jax.random.split(key_data, 2)
                init_params = {
                    "W1": jax.random.normal(init_key1, (k1, d), dtype=jnp.float32),
                    "v": jax.random.normal(init_key2, (k1,), dtype=jnp.float32)
                }
        else:
            if info_init:
                init_params = {"W1": W1_teacher}
            else:
                init_key1 = key_data
                init_params = {
                    "W1": jax.random.normal(init_key1, (k1, d), dtype=jnp.float32)
                }

        nuts_kernel = NUTS(
            model,
            adapt_step_size=True,
            target_accept_prob=acc_rate,
            dense_mass=False,
            max_tree_depth=tree_depth,
        )

        # MCMC: draw S posterior samples
        num_posterior_samples = 500
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup_samples, num_samples=num_posterior_samples, progress_bar=True)

        mcmc.run(key_mcmc, X, y=y, init_params=init_params)

        samples = mcmc.get_samples()
        W1_samples = samples["W1"]    # (S, k1, d)
        S = int(W1_samples.shape[0])

        if lr:
            v_samples = samples["v"]  # (S, k1)
        else:
            v_samples = jnp.broadcast_to(jnp.asarray(v_teacher)[None, :], (S, k1))

        # choose last L samples to save diagnostics
        L = min(40, S)
        last_W1 = W1_samples[-L:, :, :]    # (L, k1, d)
        last_v  = v_samples[-L:, :]        # (L, k1)

        # diag_W1_last: (L, k1)  - overlap per hidden neuron
        diag_W1_last = jnp.sum(last_W1 * W1_teacher[None, :, :], axis=2) / d

        # diag_W1_feature_last: (L, d) - feature-space diagonals
        diag_W1_feature_last = jnp.sum(last_W1 * W1_teacher[None, :, :], axis=1) / k1

        # ---------------------------
        # Compute R2
        # S2_teacher = (1/sqrt(k1)) * W1_teacher^T diag(v_teacher) W1_teacher  (d x d)
        C_t = (W1_teacher * jnp.asarray(v_teacher)[:, None])              # (k1, d)
        S2_teacher = (1.0 / jnp.sqrt(k1)) * (C_t.T @ W1_teacher)          # (d, d)

        # Samples: C_samples = last_W1 * last_v[:,:,None] -> (L, k1, d)
        C_samples = last_W1 * last_v[:, :, None]                         # (L, k1, d)
        S2_samples = (1.0 / jnp.sqrt(k1)) * jnp.matmul(C_samples.transpose(0, 2, 1), last_W1)  # (L, d, d)

        inner = jnp.sum(S2_samples * S2_teacher[None, :, :], axis=(1, 2))  # (L,)
        R2_j = jnp.mean(inner) / (float(d) * float(d))
        R2 = float(R2_j)

        # v_teacher repeated L times for saving
        v_teacher_j = jnp.asarray(v_teacher, dtype=jnp.float32)   # (k1,)
        v_teacher_last = jnp.broadcast_to(v_teacher_j[None, :], (L, k1))  # (L, k1)

        # s_sample is student readout samples
        s_sample_last = last_v   # (L, k1)

        # Save last single posterior sample 
        last_W1_sample_np = np.asarray(W1_samples[-1])[None, :, :]  # (1, k1, d)
        last_v_sample_np  = np.asarray(v_samples[-1])[None, :]      # (1, k1)
        W1_teacher_store_np = np.asarray(W1_teacher)[None, :, :]    # (1, k1, d)
        v_teacher_store_np  = np.asarray(v_teacher_j)[None, :]      # (1, k1)

        # --- Gibbs error: compute per-sample test MSE in chunks and then average over S ---
        X_test = jax.random.normal(key_test, (n_test, d), dtype=jnp.float32)
        y_test = teacher_forward(X_test)

        X_test = jax.device_put(X_test)
        y_test = jax.device_put(y_test)
        n_test = X_test.shape[0]

        # batched predictor for a chunk Xb returning (S, B)
        def predict_chunk_all(W1s, vs, Xb):
            def single_pred(W1, v):
                h1 = activation((1.0/jnp.sqrt(d)) * (Xb @ W1.T))    # (B, k1)
                return (1.0/jnp.sqrt(k1)) * (h1 @ v)                # (B,)
            return jax.vmap(single_pred)(W1s, vs)

        sqerr_acc = jnp.zeros((S,), dtype=jnp.float32)
        test_chunk = 2000
        for i0 in range(0, n_test, test_chunk):
            i1 = min(n_test, i0 + test_chunk)
            Xb = X_test[i0:i1]
            yb = y_test[i0:i1]
            preds_s = predict_chunk_all(W1_samples, v_samples, Xb)   # (S, B)
            se_chunk = jnp.sum((preds_s - yb[None, :]) ** 2, axis=1)  # (S,)
            sqerr_acc = sqerr_acc + se_chunk

        gibbs_error_per_sample = sqerr_acc / n_test   # (S,)
        gibbs_error_mean = jnp.mean(gibbs_error_per_sample)  # scalar
        teacher_error_mean = float(gibbs_error_mean) / 2.0

        results[alpha].append(teacher_error_mean)
        print(f"[proc {_proc_idx}] Alpha: {alpha} | Teacher {t+1}/{num_teacher}: Training samples: {n:8d}, Gibbs error/2 (mean over {S} post samples): {teacher_error_mean:.5f}")

        save_results_to_csv(results, results_csv_filename, num_teacher)

        diag_W1_np = np.asarray(diag_W1_last)              # (L, k1)
        diag_W1_feature_np = np.asarray(diag_W1_feature_last)  # (L, d)
        v_teacher_np = np.asarray(v_teacher_last)          # (L, k1)
        s_sample_np = np.asarray(s_sample_last)            # (L, k1)
        R2_np = np.asarray([R2], dtype=np.float32)
        alpha_np = np.asarray([alpha], dtype=np.float32)
        teacher_idx_np = np.asarray([_proc_idx] * L, dtype=np.int32)
        gibbs_error_np = np.asarray([float(gibbs_error_mean)], dtype=np.float32)

        new_entry = dict(
            diag_W1 = diag_W1_np,
            diag_W1_feature = diag_W1_feature_np,
            v_teacher = v_teacher_np,
            s_sample = s_sample_np,
            R2 = np.repeat(R2_np, L, axis=0),
            last_W1_sample = last_W1_sample_np,      # (1, k1, d)
            last_v_sample = last_v_sample_np,        # (1, k1)
            W1_teacher_store = W1_teacher_store_np,  # (1, k1, d)
            v_teacher_store = v_teacher_store_np,    # (1, k1)
            alpha = np.repeat(alpha_np, L, axis=0),      # (L,)
            teacher_idx = teacher_idx_np,
            gibbs_error = np.repeat(gibbs_error_np, L, axis=0)
        )

        save_append_single_file_1l(single_diags_filename, new_entry, k1, d)
        print(f"[proc {_proc_idx}] Appended diagnostics for last {L} posterior samples (alpha={alpha}, teacher={t+1}) into {single_diags_filename}")

        # Cleanup
        del mcmc, samples, W1_samples, v_samples
        del last_W1, last_v
        gc.collect()

print(f"\n[proc {_proc_idx}] Final overall Gibbs errors have been saved to {results_csv_filename}")
print(f"[proc {_proc_idx}] All diagnostics saved to {single_diags_filename}")