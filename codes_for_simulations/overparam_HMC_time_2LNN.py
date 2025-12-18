import os
import csv
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import math
from jax import random

# -----------------------------
# Configuration
# -----------------------------
jax.config.update("jax_enable_x64", False)

# single dimension 
d_vals = [200]

gamma1 = 0.5            # k1/d
alpha = 4.0             # n/d^2
delta = 0.03             # noise variance
sigma_const = math.sqrt(delta)
info_init = 0               # 0: random init, 1: info init
quenched = 1               # 1: quenched, 0: annealed
lr = 0                      # 0: fixed v, 1: learnable v

num_warmup_samples = 150        
num_posterior_samples = 200
points = 300           

tree_depth = 7              # max tree depth for NUTS
n_test = 10000              # number of test points

# run 10 teacher instances
num_teachers = 10

# run 10 independent HMC chains per teacher (different random initializations)
num_chains = 10

# s values to average over 
s_vals = [1, 2, 5, 10]

# -----------------------------
# Helpers
# -----------------------------
def activation(x):
    return jax.nn.relu(x)

def make_teacher_forward(W1_teacher, v_teacher, d, k1):
    def teacher_forward(x):
        h1 = activation((1/jnp.sqrt(d)) * x @ W1_teacher.T)
        return (1/jnp.sqrt(k1)) * h1 @ v_teacher
    return teacher_forward

def student_predict(W1, X, d, k1, v):
    h1 = activation((1/jnp.sqrt(d)) * X @ W1.T)
    return (1/jnp.sqrt(k1)) * h1 @ v

def read_done_teachers_from_csv(csv_name, current_d, gamma1):
    """Return set of teacher indices already present in csv for this d."""
    done = set()
    if not os.path.exists(csv_name):
        return done
    try:
        with open(csv_name, newline='') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                try:
                    d_val_s = row.get('d', '')
                    t_idx_s = row.get('teacher_idx', row.get('teacher_id', ''))
                    if d_val_s is None or d_val_s == '':
                        raise ValueError("missing 'd'")
                    if t_idx_s is None or t_idx_s == '':
                        raise ValueError("missing 'teacher_idx/teacher_id'")
                    d_val = int(float(d_val_s))
                    t_idx = int(float(t_idx_s))
                except Exception as e:
                    print(f"Skipping existing CSV row {row_idx} due to missing d or teacher_idx: {e}")
                    continue

                if d_val != current_d:
                    continue

                k1_row = int(gamma1 * d_val)
                if k1_row <= 0 or d_val <= 0:
                    print(f"Skipping existing CSV row {row_idx} with non-positive dimensions d={d_val}, k1={k1_row}")
                    continue

                done.add(int(t_idx))
    except Exception as e:
        print(f"Could not read existing CSV {csv_name}: {e}. Will re-run all teachers for d={current_d}.")
        done = set()
    return done

def append_rows_to_csv(csv_name, rows, fieldnames):
    """Append rows (list of dicts) to csv_name. If file does not exist, create and write header."""
    write_header = not os.path.exists(csv_name)
    with open(csv_name, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

# -----------------------------
# Main loop over dimensions
# -----------------------------
master_seed = 2
master_key = jax.random.PRNGKey(master_seed)

for idx, d in enumerate(d_vals):
    print(f"\n--- Running for d = {d} ({idx+1}/{len(d_vals)}) ---")

    k1 = int(gamma1 * d)
    n = int(alpha * d**2)
    sigma = sigma_const

    # CSV filename for this dimension
    csv_name = f'OVERPARAM_errors_ones_relu_d{d}.csv'

    done_teachers = read_done_teachers_from_csv(csv_name, d, gamma1)
    if done_teachers:
        print(f"Found existing entries for d={d}, skipping teacher indices: {sorted(done_teachers)}")

    rows_to_append = []

    for t in range(num_teachers):
        if t in done_teachers:
            print(f"[d={d}] Teacher {t} already present in {csv_name}, skipping.")
            continue

        # split fresh RNGs for this teacher instance
        master_key, key_teacher, key_data, key_test = jax.random.split(master_key, 4)

        key_w1 = key_teacher
        W1_teacher = jax.random.normal(key_w1, shape=(k1, d)).astype(jnp.float64)
        v_teacher = jnp.ones((k1,), dtype=jnp.float64)

        # generate training data (use independent keys for X and noise)
        key_X, key_noise, _ = jax.random.split(key_data, 3)
        X = jax.random.normal(key_X, (n, d), dtype=jnp.float64)
        def teacher_forward_data(x):
            h1 = activation((1/jnp.sqrt(d)) * x @ W1_teacher.T)
            return (1/jnp.sqrt(k1)) * h1 @ v_teacher
        y = teacher_forward_data(X) + sigma * jax.random.normal(key_noise, (n,), dtype=jnp.float64)

        # Student probabilistic model: W1 is sampled; v is fixed to ones (not learnable)
        v_fixed = jnp.ones((k1,), dtype=jnp.float64)

        def model(X, y=None):
            W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([k1, d]).to_event(2))
            h1 = activation((1/jnp.sqrt(d)) * X @ W1.T)
            f = (1/jnp.sqrt(k1)) * h1 @ v_fixed
            with numpyro.plate("obs", X.shape[0]):
                numpyro.sample("y", dist.Normal(f, sigma), obs=y)

        # test set 
        teacher_forward = make_teacher_forward(W1_teacher, v_teacher, d, k1)
        X_test = jax.random.normal(key_test, (n_test, d))
        y_test = teacher_forward(X_test)

        # For this teacher, run num_chains independent MCMC warmups with different uninformative inits
        preds_per_chain = []  # will collect (num_points, n_test) per chain
        iterations = None

        print(f"[d={d}] Teacher {t+1}/{num_teachers}: running {num_chains} independent HMC chains ...")
        for chain_idx in range(num_chains):
            master_key, key_chain_init, key_chain_run = jax.random.split(master_key, 3)

            # uninformative random init for this chain (W1 random normal)
            key_init_W1 = key_chain_init
            init_params = {'W1': jax.random.normal(key_init_W1, (k1, d), dtype=jnp.float64)}

            # instantiate and run MCMC (collect warmup)
            nuts_kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.65,
                               dense_mass=False, max_tree_depth=tree_depth)
            mcmc = MCMC(nuts_kernel, num_warmup=num_warmup_samples, num_samples=num_posterior_samples)

            print(f"[d={d}] Teacher {t}, Chain {chain_idx+1}/{num_chains}: warmup (collect_warmup=True)")
            mcmc.warmup(key_chain_run, X, y=y, init_params=init_params, collect_warmup=True)

            samples = mcmc.get_samples()
            if 'W1' not in samples:
                raise RuntimeError("W1 not found in MCMC samples. Check NumPyro version or warmup collection.")

            samples_W1 = samples['W1']  # warmup/posterior samples for W1

            # subsample indices across available samples
            num_samples_available = samples_W1.shape[0]
            step = max(1, int(num_samples_available / points))
            indices = list(range(0, num_samples_available, step))

            if iterations is None:
                iterations = [int(tree_depth * (i + 1)) for i in range(len(indices))]

            # compute predictions along these indices using v_teacher (as requested)
            preds_this_chain = []
            for i in indices:
                W1_sample = samples_W1[i]
                v_sample = v_teacher  # per user instruction: use v_teacher for predictions
                pred_test = student_predict(W1_sample, X_test, d, k1, v_sample)
                preds_this_chain.append(np.asarray(pred_test))

            preds_this_chain = np.stack(preds_this_chain, axis=0)  # (num_points, n_test)
            preds_per_chain.append(preds_this_chain)

            print(f"[d={d}] Teacher {t}, Chain {chain_idx+1}/{num_chains}: collected {preds_this_chain.shape[0]} points.")

        # stack chains -> shape (num_chains, num_points, n_test)
        preds_per_chain = np.stack(preds_per_chain, axis=0)
        num_points = preds_per_chain.shape[1]
        print(f"[d={d}] Teacher {t}: collected predictions shape across chains: {preds_per_chain.shape}")

        # compute averaged predictions over first s chains and compute MSE
        for point_idx in range(num_points):
            iter_val = iterations[point_idx]
            for s in s_vals:
                if s > num_chains:
                    continue
                avg_pred = np.mean(preds_per_chain[:s, point_idx, :], axis=0)  # (n_test,)
                mse_err = float(np.mean((np.asarray(y_test) - avg_pred)**2))  

                rows_to_append.append({
                    'd': str(int(d)),
                    'iteration': str(int(iter_val)),
                    'teacher_idx': str(int(t)),
                    's': str(int(s)),
                    'gibbs_error': repr(float(mse_err))
                })

        print(f"[d={d}] Teacher {t+1}/{num_teachers}: collected {num_points * len(s_vals)} rows (s in {s_vals}).")

    if rows_to_append:
        fieldnames = ['d', 'iteration', 'teacher_idx', 's', 'gibbs_error']
        append_rows_to_csv(csv_name, rows_to_append, fieldnames)
        print(f"Appended {len(rows_to_append)} new rows to {csv_name}.")
    else:
        print(f"No new teacher runs for d={d}; {csv_name} unchanged.")
