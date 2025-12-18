import csv
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import math
from jax import random
import scipy.integrate   
import gc
import time
import json

BASE_SEED = 1234
rng_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", False)
devices = jax.devices()
print(f"[proc {_proc_idx}] Visible devices for this process:", devices)

# Hyperparameters.
d = 200                   # Input dimension (k0)
gamma1 = 0.5              # k1/d
gamma2 = 0.5              # k2/d
#alphas = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
alphas = [0.25, 0.75, 1.25, 1.75, 2.15, 2.35, 2.55, 2.75, 2.95, 3.15, 3.35, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25]
delta = 0.2                 # noise variance
sigma = jnp.sqrt(delta)
info_init = 1           # whether to use informative initialization or random initialization
quenched = 0            # whether to use quenched v quenched or not
lr = 1                  # whether to use learnable readouts or fixed readouts
tree_depth = 8          # max tree depth for NUTS

num_teacher = 20        # number of teacher realizations per alpha

acc_rate = 0.6          # target acceptance rate for NUTS

n_test = 100000         # number of test samples for Gibbs error estimation

# ---------- Activation ----------
const_lin = 0.729477531486114827430355944671
var_tanh_nl, _ = scipy.integrate.quad(
    lambda x: (np.tanh(2*x) - const_lin * x)**2 * np.exp(-x**2/2) / (np.sqrt(2*np.pi)),
    -10, 10
)
var_tanh, error = scipy.integrate.quad(lambda x: (np.tanh(2*x))**2 *np.exp(-x**2/2)/(np.sqrt(2*np.pi)), -10, 10)

def activation(x):
    return (jax.nn.tanh(2*x))/np.sqrt(var_tanh)
# ---------------------------------

# Teacher forward function generator for 3-layer network.
def make_teacher_forward(W1_teacher, W2_teacher, v_teacher, d, k1, k2):
    def teacher_forward(x):
        h1 = activation((1/jnp.sqrt(d)) * x @ W1_teacher.T)
        h2 = activation((1/jnp.sqrt(k1)) * h1 @ W2_teacher.T)
        return (1/jnp.sqrt(k2)) * h2 @ v_teacher
    return teacher_forward

###########################
# Overall Results Storage #
###########################
results_csv_filename = f"tanh_2_info_gauss_gibbs_errors_d=200_delta02.csv"
single_diags_filename = f"sample_2_info_diags_gauss_d=200.npz"

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
def _init_empty_storage(k1, k2, d):
    return dict(
        diag_W1 = np.zeros((0, k1), dtype=np.float32),
        diag_W2 = np.zeros((0, k2), dtype=np.float32),
        diag_cross = np.zeros((0, k2), dtype=np.float32),
        diag_W1_feature = np.zeros((0, d), dtype=np.float32),
        diag_W2_feature = np.zeros((0, k1), dtype=np.float32),
        v_teacher = np.zeros((0, k2), dtype=np.float32),
        v_sample = np.zeros((0, k2), dtype=np.float32),
        v_sample_dot_W2 = np.zeros((0, k1), dtype=np.float32),
        v_teacher_dot_W2 = np.zeros((0, k1), dtype=np.float32),
        last_W1 = np.zeros((0, k1, d), dtype=np.float32),
        last_W2 = np.zeros((0, k2, k1), dtype=np.float32),
        alpha = np.zeros((0,), dtype=np.float32),
        teacher_idx = np.zeros((0,), dtype=np.int32),
        gibbs_error = np.zeros((0,), dtype=np.float32),
    )

def save_append_single_file(filename, new_entry, k1, k2, d):
    if os.path.exists(filename):
        with np.load(filename, allow_pickle=False) as data:
            stored = {k: data[k] for k in data.files}
        storage = {}
        defaults = _init_empty_storage(k1, k2, d)
        for key in defaults.keys():
            storage[key] = stored.get(key, defaults[key])
        for k in new_entry:
            storage[k] = np.concatenate((storage[k], new_entry[k]), axis=0)
    else:
        storage = _init_empty_storage(k1, k2, d)
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
    if (alpha < 1.5):
        num_warmup_samples = 15000   # warmup iterations 
    elif (alpha < 4):
        num_warmup_samples = 25000   # warmup iterations 
    elif (alpha < 9):
        num_warmup_samples = 7000   # warmup iterations 
    if alpha not in results:
        results[alpha] = []
    current_done = len(results[alpha])
    if current_done >= num_teacher:
        print(f"[proc {_proc_idx}] Alpha {alpha} already has {current_done} teacher realizations. Skipping...")
        continue

    # Compute number of training samples for the current alpha.
    n = int(alpha * d**2)

    for t in range(current_done, num_teacher):
        rng_key, key_teacher, key_data, key_mcmc, key_test = jax.random.split(rng_key, 5)
        # split teacher key for W1,W2,v,X,noisy draws
        key_W1, key_W2, key_v, key_X, key_noise = jax.random.split(key_teacher, 5)

        # Teacher network parameters.
        k1 = int(gamma1 * d)
        k2 = int(gamma2 * d)

        W1_teacher = jax.random.normal(key_W1, (k1, d), dtype=jnp.float32)
        W2_teacher = jax.random.normal(key_W2, (k2, k1), dtype=jnp.float32)

        # Generate teacher readouts
        #v_teacher = np.ones(k2)
        #v_teacher = np.mean(np.sort(np.random.randn(1000, k2), axis=1), axis=0)
        v_teacher = jax.random.normal(key_teacher, (k2,), dtype=jnp.float32)

        # Create teacher forward function (3-layer)
        teacher_forward = make_teacher_forward(W1_teacher, W2_teacher, v_teacher, d, k1, k2)

        # Generate synthetic training data.
        X = jax.random.normal(key_X, (n, d), dtype=jnp.float32)
        y = teacher_forward(X) + sigma * jax.random.normal(key_noise, (n,), dtype=jnp.float32)

        # Build local model for student (3-layer)
        if lr:
            # Learnable readouts (v) AND learnable W2
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([k1, d]).to_event(2))
                W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([k2, k1]).to_event(2))
                if quenched:
                    v = numpyro.sample("v", dist.Normal(jnp.array(v_teacher), 0.0001).expand([k2]).to_event(1))
                else:
                    v = numpyro.sample("v", dist.Normal(0, 1).expand([k2]).to_event(1))
                h1 = activation((1/jnp.sqrt(d)) * X @ W1.T)
                h2 = activation((1/jnp.sqrt(k1)) * h1 @ W2.T)
                f = (1/jnp.sqrt(k2)) * h2 @ v
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)
        else:
            # Fixed readouts (v_teacher), but still learn W1 and W2
            def model(X, y=None):
                W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([k1, d]).to_event(2))
                W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([k2, k1]).to_event(2))
                h1 = activation((1/jnp.sqrt(d)) * X @ W1.T)
                h2 = activation((1/jnp.sqrt(k1)) * h1 @ W2.T)
                f = (1/jnp.sqrt(k2)) * h2 @ v_teacher
                with numpyro.plate("obs", X.shape[0]):
                    numpyro.sample("y", dist.Normal(f, sigma), obs=y)

        if lr:
            if info_init:
                init_params = {"W1": W1_teacher, "W2": W2_teacher, "v": v_teacher}
            else:
                init_key1, init_key2, init_key3 = jax.random.split(key_data, 3)
                init_params = {
                    "W1": jax.random.normal(init_key1, (k1, d), dtype=jnp.float32),
                    "W2": jax.random.normal(init_key2, (k2, k1), dtype=jnp.float32),
                    "v":  jax.random.normal(init_key3, (k2,), dtype=jnp.float32)
                }
        else:
            if info_init:
                init_params = {"W1": W1_teacher, "W2": W2_teacher}
            else:
                init_key1, init_key2 = jax.random.split(key_data, 2)
                init_params = {
                    "W1": jax.random.normal(init_key1, (k1, d), dtype=jnp.float32),
                    "W2": jax.random.normal(init_key2, (k2, k1), dtype=jnp.float32)
                }

        nuts_kernel = NUTS(
            model,
            adapt_step_size    = True,
            target_accept_prob = acc_rate,
            dense_mass         = False,
            max_tree_depth     = tree_depth,
        )

        # ------------------------------
        # draw S posterior samples
        # ------------------------------
        num_posterior_samples = 500  # S
        mcmc = MCMC(nuts_kernel,
                    num_warmup=num_warmup_samples,
                    num_samples=num_posterior_samples,
                    progress_bar=True)

        # Run warmup + S posterior draws.
        mcmc.run(key_mcmc, X, y=y, init_params=init_params)

        # Retrieve posterior samples (arrays shaped (S, ...))
        samples = mcmc.get_samples()

        # Device arrays for samples (still on device). Shapes:
        #   W1_samples: (S, k1, d)
        #   W2_samples: (S, k2, k1)
        #   v_samples:  (S, k2)  if lr else broadcasted
        W1_samples = samples["W1"]            # (S, k1, d)
        W2_samples = samples["W2"]            # (S, k2, k1)
        S = W1_samples.shape[0]

        if lr:
            v_samples = samples["v"]          # (S, k2)
        else:
            # fixed teacher readouts -> broadcast to S
            v_samples = jnp.broadcast_to(jnp.asarray(v_teacher)[None, :], (S, k2))

        # ---------- Vectorized diagnostics (we will only save the last L samples' diagnostics) ----------
        L = min(20, S)  # how many of the last S samples to keep for per-sample diagnostics

        # select last L posterior samples (these will be saved individually)
        last_W1 = W1_samples[-L:, :, :]   # (L, k1, d)
        last_W2 = W2_samples[-L:, :, :]   # (L, k2, k1)
        last_v  = v_samples[-L:, :]       # (L, k2)

        # Compute per-sample diagnostics for the last L samples (all on-device)
        # diag_W1_last: (L, k1)
        diag_W1_last = jnp.sum(last_W1 * W1_teacher[None, :, :], axis=2) / d

        # diag_W2_last: (L, k2)
        diag_W2_last = jnp.sum(last_W2 * W2_teacher[None, :, :], axis=2) / k1

        # diag_cross_last: (L, k2)
        A = jnp.matmul(W2_teacher, W1_teacher)  # (k2, d) constant
        B_batch_last = jnp.matmul(last_W1.transpose(0, 2, 1), last_W2.transpose(0, 2, 1))  # (L, d, k2)
        # contract A (k2,d) with B_batch_last (L,d,k2) -> (L, k2)
        diag_cross_last = jnp.einsum('kd, ldk -> lk', A, B_batch_last) / (d * k1)

        # feature-space diagonals
        diag_W1_feature_last = jnp.sum(last_W1 * W1_teacher[None, :, :], axis=1) / k1  # (L, d)
        diag_W2_feature_last = jnp.sum(last_W2 * W2_teacher[None, :, :], axis=1) / k2  # (L, k1)

        # v vectors and their dot products with W2 for last L samples
        v_teacher_j = jnp.asarray(v_teacher)                     # (k2,)
        v_teacher_last = jnp.broadcast_to(v_teacher_j[None, :], (L, k2))  # (L, k2) repeated

        v_sample_last = last_v                                   # (L, k2)
        v_sample_dot_W2_last = jnp.einsum('lk, lkm -> lm', v_sample_last, last_W2)  # (L, k1)
        v_teacher_dot_W2 = jnp.dot(v_teacher_j, W2_teacher)      # (k1,)
        v_teacher_dot_W2_last = jnp.broadcast_to(v_teacher_dot_W2[None, :], (L, k1))  # (L, k1)

        # ------------- Gibbs error (test-set MSE) averaged over all S posterior samples ----------
        # We'll compute per-sample squared error across the test set in chunks and then average over S.

        # Prepare test set
        X_test = jax.random.normal(key_test, (n_test, d), dtype=jnp.float32)
        y_test = teacher_forward(X_test)

        X_test = jax.device_put(X_test)
        y_test = jax.device_put(y_test)
        n_test = X_test.shape[0]

        # helper: batched prediction for a chunk Xb -> returns (S, B)
        def predict_chunk_for_all_samples(W1s, W2s, vs, Xb):
            # single-sample predictor for Xb (B,d) -> (B,)
            def single_pred(W1, W2, v):
                h1 = activation((1.0 / jnp.sqrt(d)) * (Xb @ W1.T))    # (B, k1)
                h2 = activation((1.0 / jnp.sqrt(k1)) * (h1 @ W2.T))   # (B, k2)
                return (1.0 / jnp.sqrt(k2)) * (h2 @ v)                # (B,)

            return jax.vmap(single_pred)(W1s, W2s, vs)

        # Accumulator for squared errors per sample
        sqerr_acc = jnp.zeros((S,), dtype=jnp.float32)

        test_chunk = 2000   # adjust if you need more/less memory
        for i0 in range(0, n_test, test_chunk):
            i1 = min(n_test, i0 + test_chunk)
            Xb = X_test[i0:i1]               # (B, d)
            yb = y_test[i0:i1]               # (B,)

            preds_s = predict_chunk_for_all_samples(W1_samples, W2_samples, v_samples, Xb)  # (S, B)
            # squared error per sample on this chunk
            se_chunk = jnp.sum((preds_s - yb[None, :]) ** 2, axis=1)  # (S,)
            sqerr_acc = sqerr_acc + se_chunk

        gibbs_error_per_sample = sqerr_acc / n_test   # (S,)
        gibbs_error_mean = jnp.mean(gibbs_error_per_sample)  # scalar

        teacher_error_mean = float(gibbs_error_mean) / 2.0

        # Append overall results (mean)
        results[alpha].append(teacher_error_mean)
        print(f"[proc {_proc_idx}] Alpha: {alpha} | Teacher {t+1}/{num_teacher}: Training samples: {n:8d}, Gibbs error/2 (mean over {S} post samples): {teacher_error_mean:.5f}")

        # Save overall results CSV (per-process atomic save)
        save_results_to_csv(results, results_csv_filename, num_teacher)

        # ------------- Convert diagnostics for last L samples to NumPy for saving -------------
        diag_W1_np = np.asarray(diag_W1_last)            # (L, k1)
        diag_W2_np = np.asarray(diag_W2_last)            # (L, k2)
        diag_cross_np = np.asarray(diag_cross_last)      # (L, k2)

        diag_W1_feature_np = np.asarray(diag_W1_feature_last)   # (L, d)
        diag_W2_feature_np = np.asarray(diag_W2_feature_last)   # (L, k1)

        v_teacher_np = np.asarray(v_teacher_last)      # (L, k2)
        v_sample_np = np.asarray(v_sample_last)        # (L, k2)
        v_sample_dot_W2_np = np.asarray(v_sample_dot_W2_last)   # (L, k1)
        v_teacher_dot_W2_np = np.asarray(v_teacher_dot_W2_last) # (L, k1)

        last_W1_np = np.asarray(last_W1)               # (L, k1, d)
        last_W2_np = np.asarray(last_W2)               # (L, k2, k1)

        alpha_np = np.asarray([alpha], dtype=np.float32)
        teacher_idx_np = np.asarray([_proc_idx] * L, dtype=np.int32)  # (L,)
        gibbs_error_np = np.asarray([float(gibbs_error_mean)])       # store the averaged gibbs error as single scalar (1,)

        new_entry = dict(
            diag_W1 = diag_W1_np,
            diag_W2 = diag_W2_np,
            diag_cross = diag_cross_np,
            diag_W1_feature = diag_W1_feature_np,
            diag_W2_feature = diag_W2_feature_np,
            v_teacher = v_teacher_np,
            v_sample = v_sample_np,
            v_sample_dot_W2 = v_sample_dot_W2_np,
            v_teacher_dot_W2 = v_teacher_dot_W2_np,
            last_W1 = last_W1_np,
            last_W2 = last_W2_np,
            alpha = np.repeat(alpha_np, L, axis=0),         # (L,)
            teacher_idx = teacher_idx_np,                   # (L,)
            gibbs_error = np.repeat(gibbs_error_np, L, axis=0)  # (L,) -> store mean gibbs error repeated for each of the L rows
        )

        save_append_single_file(single_diags_filename, new_entry, k1, k2, d)
        print(f"[proc {_proc_idx}] Appended diagnostics for last {L} posterior samples (alpha={alpha}, teacher={t+1}) into {single_diags_filename}")

        # Free large objects and collect garbage
        del mcmc
        del samples
        del W1_samples, W2_samples, v_samples
        del last_W1, last_W2, last_v
        gc.collect()

print(f"\n[proc {_proc_idx}] Final overall Gibbs errors have been saved to {results_csv_filename}")
print(f"[proc {_proc_idx}] All diagnostics saved to {single_diags_filename}")